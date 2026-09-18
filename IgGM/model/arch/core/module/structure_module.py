"""
structure_module.py
-------------------
Synchronized antibody rigid-body denoising and CDR loop-local all-atom
denoising.  CDR predictions stay in the diffuser-defined anchor-local frame;
the current predicted FR coordinates are used to map those local predictions
back to global atom14 coordinates for iterative structure updates.
"""

from __future__ import annotations

import torch
from torch import nn

from IgGM.protein import ProtStruct, ProtConverter, AtomMapper
from .head import PLDDTHead
from .fr_cdr_blocks import FRBranch, CDRFusionBlock
from .simple_sfeatnse_uodate import LiteXtStructAttention


class StructureModule(nn.Module):
    """synchronized FR rigid + CDR local diffusion branches."""

    def __init__(
            self,
            n_lyrs=8,
            n_dims_sfea=384,
            n_dims_pfea=256,
            n_dims_encd=64,
            pred_oxyg=False,
            pred_schn=False,
            max_loop_positions=64,
    ):
        super().__init__()
        self.n_lyrs = n_lyrs
        self.n_dims_sfea = n_dims_sfea
        self.n_dims_pfea = n_dims_pfea
        self.n_dims_encd = n_dims_encd
        self.pred_oxyg = pred_oxyg
        self.pred_schn = pred_schn
        self.max_loop_positions = max_loop_positions

        self.activation_checkpoint = False
        self.activation_checkpoint_fn = torch.utils.checkpoint.checkpoint

        self.atom_mapper = AtomMapper()
        self.atom_set = 'fa' if self.pred_schn else ('b4' if self.pred_oxyg else 'b3')

        self.net = nn.ModuleDict()
        self.net['norm_s'] = nn.LayerNorm(self.n_dims_sfea)
        self.net['norm_p'] = nn.LayerNorm(self.n_dims_pfea)
        self.net['linear_s'] = nn.Linear(self.n_dims_sfea, self.n_dims_sfea)

        self.net['percpt_xt'] = LiteXtStructAttention(
            c_s=self.n_dims_sfea,
            c_z=self.n_dims_pfea,
            n_heads=8,
            n_atom=14,
            rbf_bins=32,
            dropout=0.1,
            use_cdr_atom=True,
        )

        self.net['plddt'] = PLDDTHead(c_s=self.n_dims_sfea)
        self.net["fr_branch"] = FRBranch(
            c_s=self.n_dims_sfea,
            c_z=self.n_dims_pfea,
        )
        self.net["cdr_fusion_block"] = CDRFusionBlock(
            c_s=self.n_dims_sfea,
            c_z=self.n_dims_pfea,
            max_positions=self.max_loop_positions,
        )

    def forward(
            self,
            aa_seqs,
            sfea_tns,
            pfea_tns,
            encd_tns,
            n_lyrs=-1,
            cord_tns_init=None,
            cmsk_tns_init=None,
            rmsk_vec_motf=None,
            chunk_size=None,
            region_metadata=None,
        ):
            # 1. Initialization and feature preparation
            n_smpls, n_resds, _ = sfea_tns.shape
            dtype, device = sfea_tns.dtype, sfea_tns.device
            n_lyrs = self.n_lyrs if n_lyrs == -1 else n_lyrs

            if not all(len(seq) == n_resds for seq in aa_seqs):
                raise ValueError("Sequence and residue feature lengths differ")

            sfea_tns_init = self.net["norm_s"](sfea_tns)
            pfea_tns = self.net["norm_p"](pfea_tns)
            sfea_tns = self.net["linear_s"](sfea_tns_init)

            curr_coords = cord_tns_init.detach().float().clone()
            curr_cmsk = cmsk_tns_init.detach().clone()

            cord_list, plddt_list, loop_cords = [], [], []
            trsl_list, rota_list, clean_label_list = [], [], []
            trsl_residual_list, rota_vec_norm_list = [], []

            # 2. Extract and expand base masks from region_metadata
            antibody_mask = self._expand_batch_mask(
                region_metadata["antibody_mask"].to(device=device, dtype=torch.bool), n_smpls
            )
            antigen_mask_src = region_metadata.get("antigen_mask", ~region_metadata["antibody_mask"].to(torch.bool))
            antigen_mask = self._expand_batch_mask(
                antigen_mask_src.to(device=device, dtype=torch.bool), n_smpls
            )
            fr_mask = self._expand_batch_mask(
                region_metadata["fr_mask"].to(device=device, dtype=torch.bool), n_smpls
            )
            cdr_mask = self._expand_batch_mask(
                region_metadata["cdr_mask"].to(device=device, dtype=torch.bool), n_smpls
            )

            # 3. Extract and expand loop-specific metadata
            loop_type_ids = self._expand_batch_mask(region_metadata["loop_type_ids"].to(device), n_smpls)
            loop_global_res_indices = self._expand_batch_mask(region_metadata["loop_global_res_indices"].to(device), n_smpls)
            loop_valid_res_mask = self._expand_batch_mask(
                region_metadata["loop_valid_res_mask"].to(device=device, dtype=torch.bool), n_smpls
            )
            loop_atom_valid_mask = self._expand_batch_mask(
                region_metadata["loop_atom_valid_mask"].to(device=device, dtype=torch.bool), n_smpls
            )
            
            loop_atom_supervise_mask_src = region_metadata.get("loop_atom_supervise_mask", region_metadata["loop_atom_valid_mask"])
            loop_atom_supervise_mask = self._expand_batch_mask(
                loop_atom_supervise_mask_src.to(device=device, dtype=torch.bool), n_smpls
            )

            loop_left_anchor_idx = self._expand_batch_mask(region_metadata["loop_left_anchor_idx"].to(device), n_smpls)
            loop_right_anchor_idx = self._expand_batch_mask(region_metadata["loop_right_anchor_idx"].to(device), n_smpls)
            loop_true_len = self._expand_batch_mask(region_metadata["loop_true_len"].to(device), n_smpls)

            # 4. Extract and expand framework and anchor spatial parameters
            rota_xt = region_metadata["anchor_frame_meta"]["rota_xt"].to(device=device, dtype=torch.float32)
            if rota_xt.ndim == 2: rota_xt = rota_xt.unsqueeze(0)
            if rota_xt.shape[0] == 1 and n_smpls > 1: rota_xt = rota_xt.expand(n_smpls, -1, -1).contiguous()
            rota_xt0 = rota_xt.detach().clone()

            trsl_xt_physical = region_metadata["anchor_frame_meta"]["trsl_xt_physical"].to(device=device, dtype=torch.float32)
            if trsl_xt_physical.ndim == 1: trsl_xt_physical = trsl_xt_physical.unsqueeze(0)
            if trsl_xt_physical.shape[0] == 1 and n_smpls > 1: trsl_xt_physical = trsl_xt_physical.expand(n_smpls, -1).contiguous()

            antibody_local_coords = region_metadata["antibody_local_coords"].to(device=device, dtype=torch.float32)
            if antibody_local_coords.ndim == 3: antibody_local_coords = antibody_local_coords.unsqueeze(0)
            if antibody_local_coords.shape[0] == 1 and n_smpls > 1:
                antibody_local_coords = antibody_local_coords.expand(n_smpls, -1, -1, -1).clone()

            loop_xt_local = region_metadata["noisy_loop_local_coords"].to(device=device, dtype=torch.float32)
            if loop_xt_local.ndim == 4: loop_xt_local = loop_xt_local.unsqueeze(0)
            if loop_xt_local.shape[0] == 1 and n_smpls > 1:
                loop_xt_local = loop_xt_local.expand(n_smpls, -1, -1, -1, -1).clone()

            # 5. Extract diffusion schedule parameters (sigmas, scale, mu, c_in)
            fr_sigma_trsl = region_metadata["anchor_frame_meta"]["fr_sigma_trsl"].to(device, dtype=torch.float32).reshape(-1)
            if fr_sigma_trsl.numel() == 1: fr_sigma_trsl = fr_sigma_trsl.expand(n_smpls)

            fr_sigma_rota = region_metadata["anchor_frame_meta"]["fr_sigma_rota"].to(device, dtype=torch.float32).reshape(-1)
            if fr_sigma_rota.numel() == 1: fr_sigma_rota = fr_sigma_rota.expand(n_smpls)

            fr_c_in = region_metadata["anchor_frame_meta"]["fr_c_in"].to(device, dtype=torch.float32).reshape(-1)
            if fr_c_in.numel() == 1: fr_c_in = fr_c_in.expand(n_smpls)

            trsl_scale = region_metadata["anchor_frame_meta"]["trsl_scale"].to(device, dtype=torch.float32)

            cdr_sigma = region_metadata["cdr_meta"]["cdr_sigma"].to(device, dtype=torch.float32).reshape(-1)
            if cdr_sigma.numel() == 1: cdr_sigma = cdr_sigma.expand(n_smpls)

            cdr_c_in = region_metadata["cdr_meta"]["cdr_c_in"].to(device, dtype=torch.float32).reshape(-1)
            if cdr_c_in.numel() == 1: cdr_c_in = cdr_c_in.expand(n_smpls)
            cdr_c_in = cdr_c_in.view(n_smpls, 1, 1, 1, 1)

            cdr_mu = region_metadata["cdr_meta"]["cdr_mu"].to(device=device, dtype=torch.float32)
            cdr_scale = region_metadata["cdr_meta"]["cdr_scale"].to(
                device=device, dtype=torch.float32
            )

            cdr_xt_centered = region_metadata["cdr_meta"]["cdr_xt_centered"].to(device=device, dtype=torch.float32)
            if cdr_xt_centered.ndim == 4: cdr_xt_centered = cdr_xt_centered.unsqueeze(0)
            if cdr_xt_centered.shape[0] == 1 and n_smpls > 1:
                cdr_xt_centered = cdr_xt_centered.expand(n_smpls, -1, -1, -1, -1).clone()

            loop_xt_scaled = cdr_xt_centered * cdr_c_in

            fr_c_skip = region_metadata["anchor_frame_meta"]["fr_c_skip"].to(device=device,dtype=torch.float32,).reshape(-1)
            if fr_c_skip.numel() == 1:
                fr_c_skip = fr_c_skip.expand(n_smpls)
            fr_c_out = region_metadata["anchor_frame_meta"]["fr_c_out"].to(device=device,dtype=torch.float32,).reshape(-1)
            if fr_c_out.numel() == 1:
                fr_c_out = fr_c_out.expand(n_smpls)

            fr_rota_rms = region_metadata["anchor_frame_meta"]["fr_rota_rms"].to(
                device=device, dtype=torch.float32).reshape(-1)
            if fr_rota_rms.numel() == 1:
                fr_rota_rms = fr_rota_rms.expand(n_smpls)

            loop_context_local = loop_xt_local
            for layer_idx in range(n_lyrs):
                curr_coords = curr_coords.detach()

                # Global interaction & updates
                if self.activation_checkpoint:
                    sfea_tns = self.activation_checkpoint_fn(
                        self.net["percpt_xt"],
                        sfea_tns, pfea_tns, curr_coords, curr_cmsk,
                        cdr_mask, antibody_mask, antigen_mask, chunk_size,
                        use_reentrant=False,
                    )
                else:
                    sfea_tns = self.net["percpt_xt"](
                        sfea_tns=sfea_tns,
                        pfea_tns=pfea_tns,
                        curr_coords=curr_coords,
                        atom_mask=curr_cmsk,
                        cdr_mask=cdr_mask,
                        antibody_mask=antibody_mask,
                        antigen_mask=antigen_mask,
                        chunk_size=chunk_size,
                    )

                if layer_idx == 0:
                    sfea_tns_init = sfea_tns.clone()

                # Framework branch update
                fr_out = self.net["fr_branch"](
                    sfea_tns=sfea_tns,
                    sfea_tns_init=sfea_tns_init,
                    pfea_tns=pfea_tns,
                    encd_tns=encd_tns,
                    antibody_mask=antibody_mask,
                    antigen_mask=antigen_mask,
                    fr_mask=fr_mask,
                    curr_coords=curr_coords,
                    antibody_local_coords=antibody_local_coords,
                    rota_xt=rota_xt0,
                    trsl_xt_physical=trsl_xt_physical,
                    fr_c_in=fr_c_in,
                    fr_c_skip=fr_c_skip,
                    fr_c_out=fr_c_out,
                    trsl_scale=trsl_scale,
                    fr_sigma_trsl=fr_sigma_trsl,
                    fr_rota_rms=fr_rota_rms,
                )

                fr_coords = fr_out["fr_coords"]
                sfea_tns = fr_out["sfea_tns"]
                pred_trsl = fr_out["trsl"]
                pred_rota = fr_out["rota"]

                # CDR branch update & fusion
                cdr_out = self.net["cdr_fusion_block"](
                    sfea_tns_for_cdr=sfea_tns,
                    sfea_tns_init=sfea_tns_init,
                    encd_tns=encd_tns,
                    full_sfea=sfea_tns,
                    pfea_tns=pfea_tns,
                    antigen_mask=antigen_mask,
                    fr_coords=fr_coords,
                    loop_true_len=loop_true_len,
                    loop_type_ids=loop_type_ids,
                    loop_global_res_indices=loop_global_res_indices,
                    loop_valid_res_mask=loop_valid_res_mask,
                    loop_atom_valid_mask=loop_atom_supervise_mask,
                    loop_left_anchor_idx=loop_left_anchor_idx,
                    loop_right_anchor_idx=loop_right_anchor_idx,
                    loop_xt_scaled=loop_xt_scaled,
                    loop_context_local_physical=loop_context_local,
                    cdr_mu=cdr_mu,
                    cdr_scale=cdr_scale,
                    cdr_sigma=cdr_sigma,
                )

                curr_coords = cdr_out["merged_coords"]
                loop_context_local = cdr_out["pred_x0_local"].detach()
                sfea_tns = cdr_out["sfea_after_cdr"]
                plddt_dict = self.net["plddt"](sfea_tns.detach())

                # Log outputs for current layer
                cord_list.append(curr_coords.clone())
                trsl_list.append(pred_trsl.clone())
                rota_list.append(pred_rota.clone())
                loop_cords.append(cdr_out["pred_x0_local"].clone())
                plddt_list.append(plddt_dict)
                trsl_residual_list.append(fr_out["trsl_residual"].clone())
                rota_vec_norm_list.append(fr_out["rota_vec_norm"].clone())

            pi_logits = cdr_out["cdr_pred"]["pi_logits"]
            seq_logits = cdr_out["cdr_pred"].get("seq_logits")

            return (
                sfea_tns,
                cord_list,
                plddt_list,
                trsl_list,
                rota_list,
                loop_cords,
                pi_logits,
                clean_label_list,
                trsl_residual_list,
                rota_vec_norm_list,
                seq_logits,
            )

    @staticmethod
    def _expand_batch_mask(mask, n_smpls):
        return mask.unsqueeze(0).expand(n_smpls, *mask.shape)
