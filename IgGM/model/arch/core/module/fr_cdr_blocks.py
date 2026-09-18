from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .cdr_loop_head import CDRLoopHead
from IgGM.utils.fr_cdr_diffusion_utils import (
    extract_trsl_rota_from_noisefr,
    local_to_global_coords,
)
from IgGM.utils.diff_util import so3_log_vector


class FRBranch(nn.Module):
    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_e: int = 64,
        c_hidden: int = 384,
        c_pose: int = 128,
        pose_heads: int = 8,
        rbf_bins: int = 16,
    ) -> None:
        super().__init__()
        assert c_pose % pose_heads == 0

        self.pose_heads = pose_heads
        self.pose_head_dim = c_pose // pose_heads
        self.rbf_bins = rbf_bins

        self.noise_embed = nn.Sequential(
            nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 32)
        )
        self.res_proj = nn.Sequential(
            nn.LayerNorm(c_s * 2 + c_e),
            nn.Linear(c_s * 2 + c_e, c_hidden),
            nn.SiLU(),
            nn.Linear(c_hidden, c_hidden),
            nn.SiLU(),
        )
        self.pool_proj = nn.Sequential(
            nn.LayerNorm(c_hidden), nn.Linear(c_hidden, c_hidden), nn.SiLU()
        )

        self.iface_dim = 64
        self.iface_proj = nn.Sequential(
            nn.LayerNorm(c_hidden), nn.Linear(c_hidden, self.iface_dim), nn.SiLU()
        )

        self.pose_q = nn.Linear(c_hidden, c_pose, bias=False)
        self.pose_k = nn.Linear(c_hidden, c_pose, bias=False)
        self.pose_v = nn.Linear(c_hidden, c_pose, bias=False)
        self.pose_pair_bias = nn.Linear(c_z, pose_heads, bias=False)
        self.pose_dist_bias = nn.Linear(rbf_bins, pose_heads, bias=False)
        self.pose_dropout = nn.Dropout(0.1)

        
        self.rota_noise_embed = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32)
        )

        self.rota_q = nn.Linear(c_hidden, c_pose, bias=False)
        self.rota_k = nn.Linear(c_hidden, c_pose, bias=False)
        self.rota_v = nn.Linear(c_hidden, c_pose, bias=False)

        self.rota_pair_bias = nn.Linear(c_z, pose_heads, bias=False)
        self.rota_dist_bias = nn.Linear(rbf_bins, pose_heads, bias=False)
        self.rota_pair_gate = nn.Linear(c_z, pose_heads, bias=False)
        self.rota_dist_gate = nn.Linear(rbf_bins, pose_heads, bias=False)
        self.rota_query_logits = nn.Linear(c_hidden, pose_heads, bias=False)

        # rota_geom_dim = c_pose + pose_heads * 15
        rota_geom_dim = c_pose + pose_heads * 12
        self.rota_context_proj = nn.Sequential(
            nn.LayerNorm(rota_geom_dim),
            nn.Linear(rota_geom_dim, c_pose),
            nn.SiLU(),
            nn.Linear(c_pose, c_pose),
            nn.SiLU(),
        )

        import math
        self.pose_rbf_dim = rbf_bins * 2
        pose_token_dim = c_pose+ pose_heads * 3+ self.pose_rbf_dim

        self.pose_token_proj = nn.Sequential(
            nn.LayerNorm(pose_token_dim),
            nn.Linear(pose_token_dim, c_pose),
            nn.SiLU(),
            nn.Linear(c_pose, c_pose),
        )
        self.pose_pool_proj = nn.Sequential(
            nn.LayerNorm(c_pose), nn.Linear(c_pose, c_pose), nn.SiLU()
        )


        self.pose_dist_bias = nn.Linear(
            self.pose_rbf_dim,
            pose_heads,
            bias=False,
        )
    
        self.register_buffer(
            "pose_rbf_centers",
            torch.linspace(0.0, 40.0, rbf_bins),
            persistent=False,
        )
        self.register_buffer(
            "pose_far_rbf_centers",
            torch.linspace(math.log1p(40.0), math.log1p(256.0), rbf_bins),
            persistent=False,
        )
        
        self.register_buffer(
            "rota_rbf_centers",
            torch.linspace(0.0, 4.0, rbf_bins),
            persistent=False,
        )

        # The trailing +12 is `ag_feature` = the antigen canonical frame in the
        # body frame (9 matrix entries + 3 rotvec), i.e. rota_xt^T @ R_ag_global.
        #
        # Why the TRANSLATION head needs it (added 2026-09-07).  Its target is
        #     target = (orig_body - c_skip * xt_body) / c_out
        #     orig_body = (trsl_orig - ag_com) @ rota_xt
        # `trsl_orig` and `ag_com` are constants per complex, but they are read
        # out in the BODY frame, whose orientation is set by the rotation draw.
        # So the target moves with rota_xt even though the physical translation
        # is fixed -- measured spread of orig_body at fixed t was 12.97/7.83/
        # 11.76 A per axis.  Since ag_feature is linear in rota_xt and so is
        # orig_body, the target is an EXACT linear function of
        # [xt_body, ag_feature].  Measured R^2 predicting the target:
        #     xt_body alone (what this head used to get) : 0.397
        #     ag_feature alone                           : 0.043
        #     both                                       : 1.0000
        # Without ag_feature the head is asked for a quantity it cannot see, and
        # 0.397 was the ceiling -- more steps could not fix it.  This mirrors
        # exactly what docs/adr/0001 already did for the rotation head ("so the
        # network need not non-linearly invert x_t out of the pooled geometry");
        # the same argument holds verbatim for translation but had never been
        # applied.  12 extra dims against 227 is not the capacity-dilution case
        # that kept `pooled` (384 dims) out of these heads.
        trsl_input_dim = 3 + 32 + self.iface_dim + c_pose + 12
        rota_input_dim =  32 + c_pose + 12
        # PENDING SINGLE-VARIABLE ABLATION (Q22, 2026-09-05) -- trsl only.
        # `pooled` (c_hidden=384) was dropped from BOTH pose heads, but the
        # decision was only ever taken for `rota`: an isotropic pooled vector
        # cannot define a rotation axis, and it dilutes the std=1e-3 near-zero
        # init of rota_head.  Translation is a directed 3-vector and trsl_head
        # has no near-zero init, so that argument does not transfer.  Left off
        # for now because the "pooled hurts" run was at step 150 while every
        # earlier baseline was at step 75 -- that comparison is confounded.
        # Re-test as a single-variable run once S3a has a clean baseline.
        
        
        # (kept in sync with the live lines above: both now carry the +12 of
        # ag_feature, so uncommenting one of these does not break the shape)
        # trsl_input_dim = c_hidden + 3 + 32 + self.iface_dim + c_pose + 12
        # rota_input_dim = c_hidden + 32 + c_pose + 12

        self.trsl_head = nn.Sequential(
            nn.Linear(trsl_input_dim, c_hidden),
            nn.SiLU(),
            nn.Linear(c_hidden, c_hidden),
            nn.SiLU(),
            nn.Linear(c_hidden, 3),
        )
        self.rota_head = nn.Sequential(
            nn.Linear(rota_input_dim, c_hidden),
            nn.SiLU(),
            nn.Linear(c_hidden, c_hidden),
            nn.SiLU(),
            nn.Linear(c_hidden, 3),
        )
        self.delta_feat = nn.Linear(c_hidden, c_s)

        nn.init.normal_(self.trsl_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.rota_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.trsl_head[-1].bias)
        nn.init.zeros_(self.rota_head[-1].bias)

    def _rotation_rbf(self, distance: torch.Tensor) -> torch.Tensor:
        centers = self.rota_rbf_centers.to(distance)
        width = float(centers[-1] - centers[0]) / max(self.rbf_bins - 1, 1)
        return torch.exp(-((distance.unsqueeze(-1) - centers) / width) ** 2)

    @staticmethod
    def _antigen_body_frame(curr_coords, antigen_mask, rota_xt, trsl_xt):
        """Canonical frame of the (static) antigen cloud, expressed in the body frame.

        Equals rota_xt^T @ R_ag_global, so it carries the full x_t rotation
        information while staying invariant to a global rotation of the complex.
        """
        batch_size = curr_coords.shape[0]
        frames = []
        for b in range(batch_size):
            pts = curr_coords[b, antigen_mask[b]].float()
            n_ca = (pts[:, 1] - pts[:, 0]).mean(dim=0)
            ca = torch.matmul(pts[:, 1] - trsl_xt[b][None], rota_xt[b])
            guide_x = torch.matmul(n_ca[None], rota_xt[b]).squeeze(0)
            guide_y = ca[-1] - ca[0]

            centered = ca - ca.mean(dim=0)
            cov = centered.transpose(0, 1) @ centered / float(ca.shape[0])
            U, _, _ = torch.linalg.svd(cov.contiguous(), full_matrices=True)

            u0 = U[:, 0] * (1.0 if torch.dot(U[:, 0], guide_x) >= 0 else -1.0)
            g1 = U[:, 1] * (1.0 if torch.dot(U[:, 1], guide_y) >= 0 else -1.0)
            u1 = g1 - torch.dot(g1, u0) * u0
            if torch.linalg.norm(u1) < 1e-6:
                g1 = U[:, 2]
                u1 = g1 - torch.dot(g1, u0) * u0
            u1 = u1 / torch.linalg.norm(u1).clamp_min(1e-6)
            u2 = torch.cross(u0, u1, dim=-1)
            u2 = u2 / torch.linalg.norm(u2).clamp_min(1e-6)

            frame = torch.stack([u0, u1, u2], dim=-1).contiguous()
            if torch.det(frame) < 0:
                frame[:, 2] = -frame[:, 2]
            frames.append(frame)
        return torch.stack(frames, dim=0)

    def _body_rotation_context(
        self,
        res_hidden: torch.Tensor,
        pfea_tns: torch.Tensor,
        ca_coords: torch.Tensor,
        antibody_local_coords: torch.Tensor,
        antibody_mask: torch.Tensor,
        antigen_mask: torch.Tensor,
        fr_mask: torch.Tensor,
        rota_xt: torch.Tensor,
        trsl_xt: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = ca_coords.shape[:2]
        body_ca = torch.matmul(ca_coords - trsl_xt[:, None], rota_xt)
        rotation_mask = antibody_mask & fr_mask

        for b in range(batch_size):
            antibody_indices = torch.nonzero(
                antibody_mask[b], as_tuple=False
            ).squeeze(-1)
            compressed_fr_mask = fr_mask[b, antibody_indices]
            fr_indices = antibody_indices[compressed_fr_mask]
            body_ca[b, fr_indices] = antibody_local_coords[
                b, compressed_fr_mask, 1
            ].to(body_ca)

        ab_center = self._masked_mean(body_ca, rotation_mask)
        ag_center = self._masked_mean(body_ca, antigen_mask)

        ab_centered = body_ca - ab_center[:, None]
        ag_centered = body_ca - ag_center[:, None]
        ab_centered = ab_centered * rotation_mask.unsqueeze(-1).to(body_ca.dtype)
        ag_centered = ag_centered * antigen_mask.unsqueeze(-1).to(body_ca.dtype)

        ab_denom = rotation_mask.sum(-1).to(body_ca.dtype).clamp_min(1.0)
        ag_denom = antigen_mask.sum(-1).to(body_ca.dtype).clamp_min(1.0)
        ab_radius = torch.sqrt(
            ab_centered.square().sum((-1, -2)) / ab_denom
        ).clamp_min(1.0)
        ag_radius = torch.sqrt(
            ag_centered.square().sum((-1, -2)) / ag_denom
        ).clamp_min(1.0)

        # --- U1: both clouds must share ONE scale before being subtracted ------
        # Old code divided each cloud by its OWN radius, so `relative` carried a
        # per-complex scaling nuisance (measured: ag_radius/ab_radius = 0.78 on
        # 1bvk, i.e. 15.6% distortion of every entry).  Since `relative` feeds
        # both pooled_force and pooled_moment, that nuisance reached every
        # rotation feature.  ab_radius is the reference because rotation is a
        # property of the antibody body frame.
        # Old:
        #   ab_unit = ab_centered / ab_radius[:, None, None]
        #   ag_unit = ag_centered / ag_radius[:, None, None]
        ab_unit = ab_centered / ab_radius[:, None, None]
        ag_unit = ag_centered / ab_radius[:, None, None]

        relative = ag_unit[:, None, :, :] - ab_unit[:, :, None, :]
        distance = torch.linalg.norm(relative, dim=-1)
        distance_rbf = self._rotation_rbf(distance)

        q = self.rota_q(res_hidden).view(
            batch_size, seq_len, self.pose_heads, self.pose_head_dim
        ).permute(0, 2, 1, 3)
        k = self.rota_k(res_hidden).view(
            batch_size, seq_len, self.pose_heads, self.pose_head_dim
        ).permute(0, 2, 1, 3)
        v = self.rota_v(res_hidden).view(
            batch_size, seq_len, self.pose_heads, self.pose_head_dim
        ).permute(0, 2, 1, 3)

        pair_mask = (
            rotation_mask[:, None, :, None]
            & antigen_mask[:, None, None, :]
        )

        logits = torch.einsum(
            "bhid,bhjd->bhij", q, k
        ) * self.pose_head_dim ** -0.5
        logits = logits + self.rota_pair_bias(
            pfea_tns
        ).permute(0, 3, 1, 2)
        logits = logits + self.rota_dist_bias(
            distance_rbf
        ).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~pair_mask, -1e4)
        attention = torch.softmax(logits, dim=-1)
        attention = attention * rotation_mask[
            :, None, :, None
        ].to(attention.dtype)

        gate = self.rota_pair_gate(
            pfea_tns
        ).permute(0, 3, 1, 2)
        gate = gate + self.rota_dist_gate(
            distance_rbf
        ).permute(0, 3, 1, 2)
        signed_weight = attention * torch.tanh(gate)

        force = torch.einsum(
            "bhij,bijc->bhic", signed_weight, relative
        )

        matched_antigen = torch.einsum(
            "bhij,bjc->bhic",
            attention,
            ag_unit,
        )

        scalar = torch.einsum(
            "bhij,bhjd->bhid", attention, v
        )

        query_logits = self.rota_query_logits(
            res_hidden
        ).permute(0, 2, 1)
        query_logits = query_logits.masked_fill(
            ~rotation_mask[:, None], -1e4
        )
        query_weight = torch.softmax(query_logits, dim=-1)

        lever = ab_unit
        # torque = torch.cross(
        #     lever[:, None].expand_as(force),
        #     force,
        #     dim=-1,
        # )

        pooled_scalar = torch.einsum(
            "bhi,bhid->bhd", query_weight, scalar
        ).reshape(batch_size, -1)
        pooled_force = torch.einsum(
            "bhi,bhic->bhc", query_weight, force
        ).reshape(batch_size, -1)
        # pooled_torque = torch.einsum(
        #     "bhi,bhic->bhc", query_weight, torque
        # ).reshape(batch_size, -1)
        pooled_moment = torch.einsum(
            "bhi,bic,bhid->bhcd",
            query_weight,
            lever,
            matched_antigen,
        ).reshape(batch_size, -1)

        geometry = torch.cat(
            [
                pooled_scalar,
                pooled_force,
                # pooled_torque,
                pooled_moment,
            ],
            dim=-1,
        )
        
        return self.rota_context_proj(geometry)
    
    @staticmethod
    def _expand_scalar(
        value: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = value.to(device=device, dtype=dtype).reshape(-1)
        return value.expand(batch_size) if value.numel() == 1 else value

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(values.dtype)
        return (values * mask).sum(1) / mask.sum(1).clamp_min(1.0)

    @staticmethod
    def _skew_matrix(vector: torch.Tensor) -> torch.Tensor:
        x, y, z = vector.unbind(-1)
        zero = torch.zeros_like(x)
        return torch.stack(
            [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
        ).reshape(*vector.shape[:-1], 3, 3)


    @classmethod
    def _so3_exp_map(cls, vector: torch.Tensor) -> torch.Tensor:
        theta_sq = vector.square().sum(-1, keepdim=True)
        theta = theta_sq.clamp_min(1e-12).sqrt()
        theta_safe = theta.clamp_min(1e-4)

        coef_a = torch.sin(theta_safe) / theta_safe
        coef_b = (1.0 - torch.cos(theta_safe)) / theta_safe.square()
        coef_a = torch.where(
            theta_sq < 1e-8,
            1.0 - theta_sq / 6.0 + theta_sq.square() / 120.0,
            coef_a,
        )
        coef_b = torch.where(
            theta_sq < 1e-8,
            0.5 - theta_sq / 24.0 + theta_sq.square() / 720.0,
            coef_b,
        )

        skew = cls._skew_matrix(vector)
        eye = torch.eye(3, device=vector.device, dtype=vector.dtype)
        eye = eye.view(*((1,) * (vector.ndim - 1)), 3, 3)
        return eye + coef_a.unsqueeze(-1) * skew + coef_b.unsqueeze(-1) * (skew @ skew)

    # def _rbf(self, distance: torch.Tensor) -> torch.Tensor:
    #     centers = self.pose_rbf_centers.to(distance)
    #     width = float(centers[-1] - centers[0]) / max(self.rbf_bins - 1, 1)
    #     return torch.exp(-((distance.unsqueeze(-1) - centers) / width) ** 2)

    def _rbf(
        self,
        distance: torch.Tensor,
    ) -> torch.Tensor:
        near_centers = self.pose_rbf_centers.to(distance)
        near_width = float(near_centers[-1] - near_centers[0]
        ) / max(self.rbf_bins - 1,1,)
        near_rbf = torch.exp(-(( distance.unsqueeze(-1)- near_centers ) / near_width).square())

        log_distance = torch.log1p(distance.clamp_max(256.0))
        far_centers = self.pose_far_rbf_centers.to(distance)
        far_width = float( far_centers[-1] - far_centers[0]) / max(self.rbf_bins - 1,1,)

        far_rbf = torch.exp(
            -((log_distance.unsqueeze(-1)- far_centers)/ far_width).square())

        return torch.cat([ near_rbf, far_rbf,],dim=-1,)

    def _interface_pool(
        self,
        res_hidden: torch.Tensor,
        ca_coords: torch.Tensor,
        antibody_mask: torch.Tensor,
        antigen_mask: torch.Tensor,
        tau: float = 8.0,
    ) -> torch.Tensor:
        features = []
        for b in range(res_hidden.shape[0]):
            ab_idx = torch.nonzero(antibody_mask[b], as_tuple=False).squeeze(-1)
            ag_idx = torch.nonzero(antigen_mask[b], as_tuple=False).squeeze(-1)
            h_ab = res_hidden[b, ab_idx]
            distance = torch.cdist(ca_coords[b, ab_idx], ca_coords[b, ag_idx])
            weight = torch.softmax(-distance.min(-1).values / tau, dim=0).unsqueeze(-1)
            features.append((h_ab * weight).sum(0))
        return self.iface_proj(torch.stack(features))

    def _body_pose_context(
        self,
        res_hidden: torch.Tensor,
        pfea_tns: torch.Tensor,
        ca_coords: torch.Tensor,
        antibody_local_coords: torch.Tensor,
        antibody_mask: torch.Tensor,
        antigen_mask: torch.Tensor,
        fr_mask: torch.Tensor, 
        rota_xt: torch.Tensor,
        trsl_xt: torch.Tensor,
        trsl_scale: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = ca_coords.shape[:2]
        # body_ca = torch.matmul(ca_coords - trsl_xt[:, None], rota_xt)

        # for b in range(batch_size):
        #     body_ca[b, antibody_mask[b]] = antibody_local_coords[b, :, 1].to(body_ca)
        body_ca = torch.matmul(ca_coords - trsl_xt[:, None], rota_xt)

        for b in range(batch_size):
            antibody_indices = torch.nonzero(
                antibody_mask[b], as_tuple=False
            ).squeeze(-1)
            compressed_fr_mask = fr_mask[b, antibody_indices]
            fr_indices = antibody_indices[compressed_fr_mask]
            body_ca[b, fr_indices] = antibody_local_coords[
                b, compressed_fr_mask, 1
            ].to(body_ca)


        relative = body_ca[:, None] - body_ca[:, :, None]
        distance = torch.linalg.norm(relative, dim=-1)

        q = self.pose_q(res_hidden).view(
            batch_size, seq_len, self.pose_heads, self.pose_head_dim
        ).permute(0, 2, 1, 3)
        k = self.pose_k(res_hidden).view(
            batch_size, seq_len, self.pose_heads, self.pose_head_dim
        ).permute(0, 2, 1, 3)
        v = self.pose_v(res_hidden).view(
            batch_size, seq_len, self.pose_heads, self.pose_head_dim
        ).permute(0, 2, 1, 3)

        logits = torch.einsum("bhid,bhjd->bhij", q, k) * self.pose_head_dim ** -0.5
        logits = logits + self.pose_pair_bias(pfea_tns).permute(0, 3, 1, 2)
        logits = logits + self.pose_dist_bias(self._rbf(distance)).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~antigen_mask[:, None, None], -1e4)

        attention = self.pose_dropout(torch.softmax(logits, dim=-1))
        attention = attention * antibody_mask[:, None, :, None].to(attention.dtype)

        scalar_context = torch.einsum(
            "bhij,bhjd->bhid", attention, v
        ).permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        vector_context = torch.einsum(
            "bhij,bijc->bihc", attention, relative
        ).reshape(batch_size, seq_len, -1)
        vector_context = vector_context / trsl_scale[:, None, None].clamp_min(1.0)

        min_distance = distance.masked_fill(
            ~antigen_mask[:, None], 1e4
        ).min(-1).values
        token = self.pose_token_proj(
            torch.cat([scalar_context, vector_context, self._rbf(min_distance)], dim=-1)
        )
        token = token * antibody_mask.unsqueeze(-1).to(token.dtype)

        pool_logits = (-min_distance / 8.0).masked_fill(~antibody_mask, -1e4)
        pool_weight = torch.softmax(pool_logits, dim=-1).unsqueeze(-1)
        return self.pose_pool_proj((token * pool_weight).sum(1))


    def forward(
        self,
        sfea_tns: torch.Tensor,
        sfea_tns_init: torch.Tensor,
        pfea_tns: torch.Tensor,
        encd_tns: torch.Tensor,
        antibody_mask: torch.Tensor,
        antigen_mask: torch.Tensor,
        fr_mask: torch.Tensor,
        curr_coords: torch.Tensor,
        antibody_local_coords: torch.Tensor,
        rota_xt: torch.Tensor,
        trsl_xt_physical: torch.Tensor,
        fr_c_in: torch.Tensor,
        fr_c_skip: torch.Tensor,
        fr_c_out: torch.Tensor,
        trsl_scale: torch.Tensor,
        fr_sigma_trsl: torch.Tensor,
        fr_rota_rms: torch.Tensor,
    ) -> dict:
        batch_size, device, dtype = sfea_tns.shape[0], sfea_tns.device, sfea_tns.dtype

        if rota_xt.ndim == 2:
            rota_xt = rota_xt.unsqueeze(0)
        if trsl_xt_physical.ndim == 1:
            trsl_xt_physical = trsl_xt_physical.unsqueeze(0)
        if antibody_local_coords.ndim == 3:
            antibody_local_coords = antibody_local_coords.unsqueeze(0)
        if antibody_mask.ndim == 1:
            antibody_mask = antibody_mask.unsqueeze(0)
            antigen_mask = antigen_mask.unsqueeze(0)
            fr_mask = fr_mask.unsqueeze(0)

        antibody_mask = antibody_mask.bool()
        antigen_mask = antigen_mask.bool()
        fr_mask = fr_mask.bool()
        rota_xt_geom = rota_xt.to(device=device, dtype=torch.float32)
        trsl_xt_geom = trsl_xt_physical.to(device=device, dtype=torch.float32)
        rota_xt_feature = rota_xt_geom.to(dtype=dtype)
        trsl_xt_feature = trsl_xt_geom.to(dtype=dtype)

        res_hidden = self.res_proj(torch.cat([sfea_tns, sfea_tns_init, encd_tns], dim=-1))
        pooled = self.pool_proj(self._masked_mean(res_hidden, antibody_mask))
        ca_coords = curr_coords[:, :, 1]
        ca_coords_feature = ca_coords.to(dtype=dtype)
        antigen_com_geom = self._masked_mean(ca_coords.float(), antigen_mask)
        interface_feature = self._interface_pool(
            res_hidden, ca_coords_feature, antibody_mask, antigen_mask
        )

        sigma_trsl = self._expand_scalar(
            fr_sigma_trsl, batch_size, device, dtype
        ).clamp_min(1e-8)
        rota_rms = self._expand_scalar(
            fr_rota_rms, batch_size, device, dtype
        ).clamp_min(1e-8)
        trsl_scale = self._expand_scalar(trsl_scale, batch_size, device, dtype)

        noise_feature = self.noise_embed(
            torch.stack(
                [0.25 * torch.log(sigma_trsl), 0.25 * torch.log(rota_rms)], dim=-1
            )
        )

        with torch.autocast(device_type=device.type, enabled=False):
            trsl_body_geom = torch.matmul(
                (trsl_xt_geom - antigen_com_geom).unsqueeze(1), rota_xt_geom
            ).squeeze(1)
        c_in = self._expand_scalar(fr_c_in, batch_size, device, dtype)
        trsl_body_scaled = trsl_body_geom.to(dtype) * c_in.unsqueeze(-1)

        pose_context = self._body_pose_context(
            res_hidden=res_hidden,
            pfea_tns=pfea_tns,
            ca_coords=ca_coords_feature,
            antibody_local_coords=antibody_local_coords,
            antibody_mask=antibody_mask,
            antigen_mask=antigen_mask,
            fr_mask=fr_mask, 
            rota_xt=rota_xt_feature,
            trsl_xt=trsl_xt_feature,
            trsl_scale=trsl_scale.to(dtype),
        )
        rotation_context = self._body_rotation_context(
            res_hidden=res_hidden,
            pfea_tns=pfea_tns,
            ca_coords=ca_coords_feature,
            antibody_local_coords=antibody_local_coords,
            antibody_mask=antibody_mask,
            antigen_mask=antigen_mask,
            fr_mask=fr_mask,
            rota_xt=rota_xt_feature,
            trsl_xt=trsl_xt_feature,
        )

        rota_noise_feature = self.rota_noise_embed(
            (0.25 * torch.log(rota_rms)).unsqueeze(-1)
        )

        # ag_feature is computed BEFORE trsl_input because both pose heads need
        # it -- see the note on trsl_input_dim.  It used to be built after
        # trsl_input and fed only to rota_input.
        with torch.autocast(device_type=device.type, enabled=False):
            ag_frame = self._antigen_body_frame(
                curr_coords, antigen_mask, rota_xt_geom, trsl_xt_geom
            )
            ag_frame_vec = so3_log_vector(ag_frame)   # 见下

        ag_feature = torch.cat(
            [ag_frame.reshape(batch_size, 9), ag_frame_vec], dim=-1
        ).to(dtype)                                                # 12 维

        trsl_input = torch.cat(
            [
                # pooled,
                trsl_body_scaled,
                noise_feature,
                interface_feature,
                pose_context,
                ag_feature,
            ],
            dim=-1,
        )

        rota_input = torch.cat(
            [
                # pooled,
                rota_noise_feature,
                rotation_context,
                ag_feature,
            ],
            dim=-1,
        )

        pred_trsl_residual = self.trsl_head(trsl_input)


        pred_rota_vec_norm = self.rota_head(rota_input)
        with torch.autocast(device_type=device.type, enabled=False):
            c_skip = self._expand_scalar(
                fr_c_skip, batch_size, device, torch.float32
            )
            c_out = self._expand_scalar(
                fr_c_out, batch_size, device, torch.float32
            )
            rota_rms_geom = self._expand_scalar(
                fr_rota_rms, batch_size, device, torch.float32
            )
            pred_trsl_body = (
                c_skip.unsqueeze(-1) * trsl_body_geom
                + c_out.unsqueeze(-1) * pred_trsl_residual.float()
            )
            pred_trsl_global = antigen_com_geom + torch.matmul(
                pred_trsl_body.unsqueeze(1), rota_xt_geom.transpose(-1, -2)
            ).squeeze(1)
            pred_delta_rota = self._so3_exp_map(
                rota_rms_geom.unsqueeze(-1) * pred_rota_vec_norm.float()
            )
            pred_rota_global = rota_xt_geom @ pred_delta_rota

            updated_coords = curr_coords.float().clone()
            for b in range(batch_size):
                antibody_indices = torch.nonzero(
                    antibody_mask[b], as_tuple=False
                ).squeeze(-1)
                moved = local_to_global_coords(
                    antibody_local_coords[b].float(),
                    pred_rota_global[b],
                    pred_trsl_global[b],
                )
                compressed_fr_mask = fr_mask[b, antibody_indices]
                updated_coords[b, antibody_indices[compressed_fr_mask]] = moved[
                    compressed_fr_mask
                ]
        
        delta_feature = self.delta_feat(pooled).unsqueeze(1)
        delta_feature = delta_feature * antibody_mask.unsqueeze(-1).to(dtype)

        return {
            "fr_coords": updated_coords,
            "sfea_tns": sfea_tns + delta_feature,
            "trsl": pred_trsl_global,
            "trsl_body": pred_trsl_body,
            "trsl_residual": pred_trsl_residual,
            "rota": pred_rota_global,
            "delta_rota": pred_delta_rota,
            "rota_vec_norm": pred_rota_vec_norm,
            "mask": antibody_mask.any(-1),
        }

class CDRFusionBlock(nn.Module):
    def __init__(self, c_s: int = 384, c_z: int = 128, max_positions: int = 64) -> None:
        super().__init__()
        self.cdr_loop = CDRLoopHead(c_s=c_s, c_z=c_z, max_positions=max_positions)
        self.loop_feedback = nn.Sequential(
            nn.LayerNorm(14 * 3), nn.Linear(14 * 3, c_s),
            nn.SiLU(), nn.Linear(c_s, c_s)
        )

    @staticmethod
    def _gather_loop_features(features: torch.Tensor, loop_global_res_indices: torch.Tensor, loop_valid_res_mask: torch.Tensor) -> torch.Tensor:
        _, n_loop, _ = loop_global_res_indices.shape
        indices = loop_global_res_indices.clamp_min(0)
        gathered = torch.gather(
            features.unsqueeze(1).expand(-1, n_loop, -1, -1),
            dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, features.shape[-1])
        )
        return gathered * loop_valid_res_mask.unsqueeze(-1).to(gathered.dtype)

    @staticmethod
    def _gather_pair_features(pair_features: torch.Tensor, query_indices: torch.Tensor, key_indices: torch.Tensor, query_mask: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, _ = pair_features.shape
        query_indices = query_indices.clamp(min=0, max=seq_len - 1)
        key_indices = key_indices.clamp(min=0, max=seq_len - 1)
        batch_index = torch.arange(batch_size, device=pair_features.device).view(batch_size, 1, 1)

        gathered = pair_features[batch_index, query_indices.unsqueeze(-1), key_indices.unsqueeze(1)]
        pair_mask = query_mask.unsqueeze(-1) & key_mask.unsqueeze(1)
        return gathered * pair_mask.unsqueeze(-1).to(gathered.dtype)

    @staticmethod
    def _local_to_global_loop_coords(coords_local: torch.Tensor, loop_frame_rota: torch.Tensor, loop_frame_trsl: torch.Tensor, loop_atom_valid_mask: torch.Tensor) -> torch.Tensor:
        rotated = torch.einsum("bnlac,bndc->bnlad", coords_local, loop_frame_rota)
        global_coords = rotated + loop_frame_trsl[:, :, None, None, :]
        return global_coords * loop_atom_valid_mask.unsqueeze(-1).to(global_coords.dtype)

    def _feedback_sfea(self, pred_x0_local: torch.Tensor, loop_global_res_indices: torch.Tensor, loop_valid_res_mask: torch.Tensor, sfea_tns: torch.Tensor) -> torch.Tensor:
        """Scatter CDR loop features back into the full structural sequence feature tensor."""
        batch_size = pred_x0_local.shape[0]
        delta = torch.zeros_like(sfea_tns)
        signal = self.loop_feedback(pred_x0_local.reshape(batch_size, pred_x0_local.shape[1], pred_x0_local.shape[2], -1))
        
        valid = loop_valid_res_mask.to(torch.bool)
        signal = signal * valid.unsqueeze(-1).to(signal.dtype)

        for batch_idx in range(batch_size):
            indices_flat = loop_global_res_indices[batch_idx].reshape(-1)
            signal_flat = signal[batch_idx].reshape(-1, signal.shape[-1])
            valid_flat = valid[batch_idx].reshape(-1) & (indices_flat >= 0)

            if valid_flat.any():
                indices_use = indices_flat[valid_flat].to(torch.long)
                signal_use = signal_flat[valid_flat].to(delta.dtype)
                delta[batch_idx].scatter_add_(
                    0, indices_use.unsqueeze(-1).expand(-1, signal_use.shape[-1]), signal_use
                )

        return sfea_tns + delta

    @staticmethod
    def _merge_fr_cdr(fr_coords: torch.Tensor, pred_loop_global: torch.Tensor, loop_global_res_indices: torch.Tensor, loop_valid_res_mask: torch.Tensor, loop_atom_valid_mask: torch.Tensor) -> torch.Tensor:
        """Merge predicted CDR loop coordinates into the framework (FR) global coordinates."""
        merged = fr_coords.clone()
        for batch_idx in range(merged.shape[0]):
            indices = loop_global_res_indices[batch_idx].to(torch.long)
            for loop_idx in range(indices.shape[0]):
                valid = loop_valid_res_mask[batch_idx, loop_idx]
                if not valid.any(): continue
                
                global_indices = indices[loop_idx, valid]
                source = pred_loop_global[batch_idx, loop_idx, valid]
                source = source * loop_atom_valid_mask[batch_idx, loop_idx, valid].unsqueeze(-1).to(source.dtype)
                merged[batch_idx, global_indices] = source
        return merged

    @staticmethod
    def _expand_cdr_scale(scale: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        scale = scale.to(device=device, dtype=torch.float32).reshape(-1)
        if scale.numel() == 1:
            scale = scale.expand(batch_size)
        elif scale.numel() != batch_size:
            raise ValueError(f"Unexpected CDR scale shape: {tuple(scale.shape)}")
        return scale.view(batch_size, 1, 1, 1, 1)

    @staticmethod
    def _expand_cdr_mean(mean: torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mean = mean.to(device=device, dtype=dtype)
        mean = mean.reshape(1, 3).expand(batch_size, -1) if mean.numel() == 3 else mean.reshape(batch_size, 3)
        return mean.view(batch_size, 1, 1, 1, 3)

    def forward(
        self, *, sfea_tns_for_cdr: torch.Tensor, sfea_tns_init: torch.Tensor, encd_tns: torch.Tensor,
        full_sfea: torch.Tensor, pfea_tns: torch.Tensor, antigen_mask: torch.Tensor, fr_coords: torch.Tensor,
        loop_true_len: torch.Tensor, loop_type_ids: torch.Tensor, loop_global_res_indices: torch.Tensor,
        loop_valid_res_mask: torch.Tensor, loop_atom_valid_mask: torch.Tensor, loop_left_anchor_idx: torch.Tensor,
        loop_right_anchor_idx: torch.Tensor, loop_xt_scaled: torch.Tensor,
        loop_context_local_physical: torch.Tensor, cdr_mu: torch.Tensor,
        cdr_scale: torch.Tensor, cdr_sigma: torch.Tensor
    ) -> dict:
        batch_size, n_loop, lmax = loop_global_res_indices.shape
        local_position_ids = torch.arange(lmax, device=sfea_tns_for_cdr.device, dtype=torch.long)
        frame_source = fr_coords.detach().float()

        with torch.autocast(device_type=fr_coords.device.type, enabled=False):
            loop_frame_rota, loop_frame_trsl = extract_trsl_rota_from_noisefr(
                frame_source, loop_global_res_indices, loop_true_len,
                loop_left_anchor_idx, loop_right_anchor_idx
            )

        loop_sfea = self._gather_loop_features(sfea_tns_for_cdr, loop_global_res_indices, loop_valid_res_mask)
        loop_sfea_init = self._gather_loop_features(sfea_tns_init, loop_global_res_indices, loop_valid_res_mask)
        loop_encd = self._gather_loop_features(encd_tns, loop_global_res_indices, loop_valid_res_mask)
        loop_sfea = loop_sfea + loop_sfea_init

        # Local to Global conversions & C_alpha extraction
        with torch.autocast(device_type=fr_coords.device.type, enabled=False):
            context_loop_global = self._local_to_global_loop_coords(
                loop_context_local_physical.float(), loop_frame_rota,
                loop_frame_trsl, loop_atom_valid_mask
            )
        loop_ca_global = context_loop_global[:, :, :, 1, :].reshape(
            batch_size, n_loop * lmax, 3
        )

        # Cross-loop properties
        flat_loop_indices = loop_global_res_indices.reshape(batch_size, n_loop * lmax)
        flat_loop_valid = loop_valid_res_mask.reshape(batch_size, n_loop * lmax)
        loop_ids = torch.arange(n_loop, device=fr_coords.device).view(1, n_loop, 1).expand(batch_size, -1, lmax).reshape(batch_size, n_loop * lmax)
        
        cross_loop_pair_mask = flat_loop_valid.unsqueeze(-1) & flat_loop_valid.unsqueeze(1) & (loop_ids.unsqueeze(-1) != loop_ids.unsqueeze(1))
        cross_loop_pair_features = self._gather_pair_features(
            pfea_tns, flat_loop_indices, flat_loop_indices, flat_loop_valid, flat_loop_valid
        )
        cross_loop_distances = torch.cdist(loop_ca_global, loop_ca_global)

        # Loop-Antigen properties
        seq_len = pfea_tns.shape[1]
        full_indices = torch.arange(seq_len, device=pfea_tns.device, dtype=torch.long).view(1, seq_len).expand(batch_size, -1)
        antigen_mask = antigen_mask.to(torch.bool)
        
        loop_antigen_pair_features = self._gather_pair_features(
            pfea_tns, flat_loop_indices, full_indices, flat_loop_valid, antigen_mask
        )
        loop_antigen_distances = torch.cdist(loop_ca_global, frame_source[:, :, 1, :])

        # CDR loop generation pass
        cdr_pred = self.cdr_loop(
            loop_sfea=loop_sfea, loop_encd=loop_encd, loop_xt_scaled=loop_xt_scaled,
            loop_type_ids=loop_type_ids, local_position_ids=local_position_ids,
            loop_valid_res_mask=loop_valid_res_mask, loop_atom_valid_mask=loop_atom_valid_mask,
            cdr_sigma=cdr_sigma, full_sfea=full_sfea, antigen_mask=antigen_mask,
            cross_loop_pair_features=cross_loop_pair_features, cross_loop_distances=cross_loop_distances,
            cross_loop_pair_mask=cross_loop_pair_mask, loop_antigen_pair_features=loop_antigen_pair_features,
            loop_antigen_distances=loop_antigen_distances,
        )

        x0_norm = cdr_pred["x0_norm"]
        with torch.autocast(device_type=fr_coords.device.type, enabled=False):
            c_scale = self._expand_cdr_scale(
                cdr_scale, batch_size, x0_norm.device
            )
            c_mu = self._expand_cdr_mean(
                cdr_mu, batch_size, x0_norm.device, torch.float32
            )
            pred_x0_local = (
                x0_norm.float() * c_scale + c_mu
            ) * loop_atom_valid_mask.unsqueeze(-1).float()
            pred_loop_global = self._local_to_global_loop_coords(
                pred_x0_local, loop_frame_rota, loop_frame_trsl,
                loop_atom_valid_mask
            )

        # Update features and coordinates
        merged_coords = self._merge_fr_cdr(
            fr_coords, pred_loop_global, loop_global_res_indices, loop_valid_res_mask, loop_atom_valid_mask
        )
        sfea_after_cdr = self._feedback_sfea(
            pred_x0_local, loop_global_res_indices, loop_valid_res_mask, sfea_tns_for_cdr
        )

        return {
            "cdr_pred": cdr_pred,
            "pred_x0_local": pred_x0_local,
            "pred_loop_global": pred_loop_global,
            "loop_frame_rota": loop_frame_rota,
            "loop_frame_trsl": loop_frame_trsl,
            "sfea_after_cdr": sfea_after_cdr,
            "merged_coords": merged_coords,
        }
