"""Protein diffusion model for amino-acid sequences & backbone structures.

Notes:
* For <RcsbMonoDataset>, it is guaranteed (by construction) that there is no non-standard residue
    types in the amino-acid sequence.
"""

import hashlib
import math

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from IgGM.protein import AtomMapper
from IgGM.protein.prot_constants import RESD_NAMES_1C
from IgGM.utils import (
    OnlineIGSO3Schedule,
    extract_clean_fr_reference,
    extract_per_loop_clean_local_coords,
    global_to_local_coords,
    local_to_global_coords,
    merge_noisy_fr_and_loops,
    prob2seq,
    ptr2ss,
    rebuild_loops_from_local_coords,
    rebuild_and_merge_loops,
    so3_scale,
    ss2ptr,
    skew2vec,
    log_rmat,
)


class Diffuser:
    """Protein diffusion model for amino-acid sequences & backbone structures."""

    ROTATION_BASE_SEED = 20250722

    @staticmethod
    def _bucket_timestep(idx_step: int) -> int:
        idx_step = int(idx_step)

        if idx_step <= 50:
            return 25
        if idx_step <= 100:
            return 75
        if idx_step <= 150:
            return 125
        return 175

    def __init__(
            self,
            n_steps=200,  # number of time steps in the diffusion process
            pert_seq=True,  # whether to perturb amino-acid sequences
            cord_scale=4.0,  # coordinate scaling factor (in Angstrom)
            occupancy_mode="joint_predict",
            rota_angle_rms_min=0.005,
            rota_schedule_gamma=1.5,
    ):
        """Constructor function."""

        # setup configurations
        self.n_steps = n_steps
        self.pert_seq = pert_seq
        self.cord_scale = cord_scale
        self.fr_noise_scale_trsl = float(1.0)
        # CDR sigma_data~24.7; coef 1.0 -> sigma_cdr up to 3x sigma_data (from-scratch denoise)
        self.cdr_local_noise_scale =  float(0.25) # 1.0

        self.rota_angle_rms_min = float(rota_angle_rms_min)
        self.rota_schedule_gamma = float(rota_schedule_gamma)
        self.haar_angle_rms = math.sqrt(math.pi ** 2 / 3.0 + 2.0)
        if self.rota_angle_rms_min <= 0 or self.rota_schedule_gamma <= 0:
            raise ValueError("rotation angle RMS minimum and schedule gamma must be positive")

        self.occupancy_mode = occupancy_mode

        self.atom_mapper = AtomMapper()
        self.resd_names = RESD_NAMES_1C  # 20 standard AA type tokens
        self.n_tokns = len(self.resd_names)
        self.mask_rate = None

        # initialize variance schedules
        self.seq_schedule = CosineSchedule(n_steps=self.n_steps, offset=0.008, beta_max=0.999)
        self.trsl_schedule = LinearSchedule(n_steps=self.n_steps, beta_min=0.01, beta_max=0.999)
        self.__build_trmat_list()

        # --- 新增: EDM (VE) 连续时间连续噪声调度 (Karras et al. 2022) ---
        self.sigma_min = 0.01
        self.sigma_max = 80
        self.rho = 3.0
        
        # sigma_t 计算 (0: clear -> n_steps: noise)
        step_indices = torch.arange(self.n_steps + 1, dtype=torch.float64) / self.n_steps
        sigmas = (self.sigma_min**(1/self.rho) + step_indices * (self.sigma_max**(1/self.rho) - self.sigma_min**(1/self.rho))) ** self.rho
        self.sigmas = sigmas.float()

        # 1. Basic Data and Statistical Preparation (Add CDR stats alongside TRSL)
        # TODO(X2 followup): these stats were computed under the OLD all-antibody-CA
        # canonical frame.  X2 switched the frame to FR-only atoms, which shifts the
        # centroid by ~2.7 A (measured on 1bvk_B_A_C), i.e. ~10% of trsl_scale.
        # Needs a rescan of the training set before multi-sample training.
        # Harmless for single-sample overfit (one constant offset).
        self.trsl_mu = torch.tensor([-0.2222, 0.9051, 0.1434], dtype=torch.float32)
        self.trsl_scale = torch.tensor(26.0823, dtype=torch.float32) # std (sigma_data)

        # Example CDR stats (Replace with your actual computed stats)
        self.cdr_mu = torch.tensor([0,0,0], dtype=torch.float32) 
        self.cdr_scale = torch.tensor(6.0, dtype=torch.float32) # std (sigma_data)

        self._rotation_rank = None
        self.__build_rotation_schedule()

        # Atom14 sequence decoder for geometry-driven sequence feedback
        from src.iggm_lightning.atom14_sync import Atom14SeqSync
        self._atom14_sync = Atom14SeqSync()

    def __build_rotation_schedule(self) -> None:
        sigma_data_pose = torch.sqrt(self.trsl_scale * (self.cdr_scale / self.cdr_local_noise_scale))
        self.sigma_data_pose = sigma_data_pose  # kept for diagnostics (_log_forced_step)
        noise_fraction = self.sigmas.double() / torch.sqrt(self.sigmas.double().square() + sigma_data_pose.double().square())
        progress = ((noise_fraction - noise_fraction[1]) / (noise_fraction[-1] - noise_fraction[1]).clamp_min(1e-12)).clamp(0.0, 1.0)
        angle_rms = torch.zeros(self.n_steps + 1, dtype=torch.float64)
        angle_rms[1:] = self.rota_angle_rms_min + (
            self.haar_angle_rms - self.rota_angle_rms_min
        ) * progress[1:].pow(self.rota_schedule_gamma)
        angle_rms[-1] = self.haar_angle_rms
        self.rota_angle_rms_schedule = angle_rms.float()
        # Keep the float64 schedule so the reverse-sampling sampler can be built
        # from the exact same numbers (see _eval_rota_sampler).
        self._rota_angle_rms_f64 = angle_rms
        self.rota_sampler = OnlineIGSO3Schedule(angle_rms, seed=56)
        # self.rota_sampler = OnlineIGSO3Schedule(angle_rms, seed=self.ROTATION_BASE_SEED)
        self._rota_sampler_eval = None

    def _sample_probabilities(self, aa_seq_orig, pmsk_vec, idxs_step, device, generator=None):
        """Sample noisy residue-type distributions; shared by legacy and fr_cdr_sync modes.

        `generator` (reverse sampling only) pins the categorical draw.  It has to
        be threaded in explicitly: prob2seq's default path uses
        Categorical.sample(), which reads the GLOBAL RNG, so the per-sample seed
        of the fixed validation protocol never reached the sequence channel.
        Training passes None and keeps the original behaviour verbatim.
        """
        trmat_ac = self.trmat_list_ac[idxs_step].to(device)
        prob_tns_orig = nn.functional.one_hot(
            torch.tensor([self.resd_names.index(x) for x in aa_seq_orig], dtype=torch.long, device=device),
            num_classes=self.n_tokns,
        ).to(torch.float32).unsqueeze(0)
        prob_tns_pert = torch.matmul(prob_tns_orig, trmat_ac)
        prob_tns_pert = nn.functional.normalize(prob_tns_pert, p=1.0, dim=2)
        prob_tns_pert = torch.where(
            pmsk_vec.view(-1, prob_tns_orig.shape[1], 1).to(torch.bool),
            prob_tns_pert,
            prob_tns_orig,
        )
        if generator is None:
            aa_seqs_pert = prob2seq(prob_tns_pert, stoc_seq=True)
        else:
            aa_seqs_pert = self._seq_from_probs(prob_tns_pert, generator)
        return prob_tns_orig, prob_tns_pert, aa_seqs_pert

    @classmethod
    def _seq_from_probs(cls, prob_tns, generator):
        """Categorical draw pinned by an explicit CPU Generator.

        Same semantics as prob2seq(stoc_seq=True), but the draw happens on CPU
        through `generator` so one seed reproduces the whole trajectory
        regardless of which device the run is on (mirrors Diffuser._randn).
        """
        probs = prob_tns.detach().float().cpu()
        n_smpl, n_resd, n_tokn = probs.shape
        idx = torch.multinomial(
            probs.reshape(-1, n_tokn), num_samples=1, generator=generator
        ).reshape(n_smpl, n_resd)
        return [
            ''.join(RESD_NAMES_1C[i] for i in idx[s].tolist())
            for s in range(n_smpl)
        ]

    @staticmethod
    def _validate_rotation(rotation: torch.Tensor, name: str) -> None:
        with torch.amp.autocast(device_type=rotation.device.type, enabled=False):
            rotation_f32 = rotation.float()
            eye = torch.eye(3, device=rotation.device, dtype=torch.float32)
            orth_error = (rotation_f32.transpose(-1, -2) @ rotation_f32 - eye).abs().amax()
            det_error = (torch.det(rotation_f32) - 1.0).abs()
            is_finite = torch.isfinite(rotation_f32).all()
        if not is_finite or orth_error > 5e-4 or det_error > 5e-4:
            raise RuntimeError(
                f"invalid {name}: orth_error={float(orth_error):.3e}, det_error={float(det_error):.3e}"
            )

    @staticmethod
    def _build_antibody_rigid_params(
        cord_tns_orig, cmsk_mat_orig, antibody_mask, fr_mask=None
    ):
        """Build rigid-body frame for antibody.

        Args:
            fr_mask: if provided, the canonical frame (rota_orig, trsl_orig) is built
                     from FR atoms only; ab_local still covers all antibody residues
                     (the CDR will be expressed in the FR-derived frame).
        """
        import contextlib

        device = cord_tns_orig.device
        out_dtype = cord_tns_orig.dtype

        ab_mask = antibody_mask.to(device=device, dtype=torch.bool)
        if ab_mask.sum() == 0:
            raise ValueError("No antibody residues found.")

        coords_ab_orig = cord_tns_orig[ab_mask]
        atom_mask_ab_orig = cmsk_mat_orig[ab_mask]

        # --- X2: frame selection restricted to FR if requested ---
        if fr_mask is not None:
            fr_mask_ab = fr_mask[ab_mask].to(device=device, dtype=torch.bool)
            if fr_mask_ab.sum() == 0:
                raise ValueError("No FR residues found; cannot build frame.")
            frame_sel = fr_mask_ab
        else:
            frame_sel = torch.ones(
                coords_ab_orig.shape[0], device=device, dtype=torch.bool
            )

        if device.type == "cuda":
            autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=False)
        else:
            autocast_ctx = contextlib.nullcontext()

        with autocast_ctx:
            coords_ab = coords_ab_orig.float()
            atom_mask_ab = atom_mask_ab_orig.float()

            ca = coords_ab[:, 1]
            ca_mask = (atom_mask_ab[:, 1] > 0.5) & frame_sel

            if ca_mask.sum() >= 3:
                valid_points = ca[ca_mask]
            else:
                valid_atom_mask = (atom_mask_ab > 0.5) & frame_sel.unsqueeze(-1)
                valid_points = coords_ab[valid_atom_mask]

            if valid_points.numel() == 0:
                raise ValueError("No valid antibody atoms found.")

            trsl_orig_f32 = valid_points.mean(dim=0)
            centered = valid_points - trsl_orig_f32

            if valid_points.shape[0] < 3 or torch.linalg.norm(centered) < 1e-6:
                rota_orig_f32 = torch.eye(3, device=device, dtype=torch.float32)
            else:
                cov = centered.transpose(0, 1).matmul(centered)
                cov = cov / float(valid_points.shape[0])
                U, S, Vh = torch.linalg.svd(cov.float().contiguous(), full_matrices=True)
                U = U.float()

                # --- 核心强制对齐：彻底消灭 180 度翻转震荡 ---
                # X2: guides must also come from FR only, else the sign
                # disambiguation still depends on CDR conformation.
                n_ca_mask = (
                    atom_mask_ab[:, 0] * atom_mask_ab[:, 1] * frame_sel.float()
                )
                n_ca = coords_ab[:, 1] - coords_ab[:, 0]

                # X轴向 N->CA 对齐
                if n_ca_mask.sum() > 0:
                    guide_x = (n_ca * n_ca_mask[:, None]).sum(dim=0) / n_ca_mask.sum().clamp_min(1.0)
                else:
                    guide_x = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)

                # Y轴向 首尾CA 对齐
                if ca_mask.sum() >= 2:
                    ca_valid = ca[ca_mask]
                    guide_y = ca_valid[-1] - ca_valid[0]
                else:
                    guide_y = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)

                sign0 = 1.0 if torch.dot(U[:, 0], guide_x).item() >= 0.0 else -1.0
                u0 = U[:, 0] * sign0

                sign1 = 1.0 if torch.dot(U[:, 1], guide_y).item() >= 0.0 else -1.0
                guide_u1 = U[:, 1] * sign1
                u1 = guide_u1 - torch.dot(guide_u1, u0) * u0
                if torch.linalg.norm(u1) < 1e-6:
                    guide_u1 = U[:, 2]
                    u1 = guide_u1 - torch.dot(guide_u1, u0) * u0
                u1 = u1 / torch.linalg.norm(u1).clamp_min(1e-6)

                u2 = torch.cross(u0, u1, dim=-1)
                u2 = u2 / torch.linalg.norm(u2).clamp_min(1e-6)

                rota_orig_f32 = torch.stack([u0, u1, u2], dim=-1).contiguous()
                if torch.det(rota_orig_f32.float()) < 0:
                    rota_orig_f32[:, 2] = -rota_orig_f32[:, 2]

            trsl_orig_f32 = trsl_orig_f32.contiguous()

        rota_orig = rota_orig_f32.to(dtype=out_dtype)
        trsl_orig = trsl_orig_f32.to(dtype=out_dtype)

        ab_local = global_to_local_coords(coords_ab_orig, rota_orig, trsl_orig)
        ab_local = ab_local * atom_mask_ab_orig.to(dtype=out_dtype).unsqueeze(-1)

        return rota_orig, trsl_orig, ab_local

    def run(self, prot_data_orig, idxs_step=None, return_time_steps=False):
        """Build a synchronized noisy state with FR rigid motion + CDR local diffusion."""
        device_type = prot_data_orig["cord"].device.type

        # idxs_step = self._bucket_timestep(idxs_step)

        # # self.rota_sampler.generator.manual_seed(56)   # ← 私有 generator，torch.manual_seed 管不到
        # # torch.manual_seed(56)

        # # --- DEBUG OVERRIDE: pin the noise bucket ------------------------------
        # # AA2: 75.  Only the mid/low-noise regime tests whether the model can
        # # actually localise the pose.  At high noise (e.g. 120 -> rota_rms 66.8deg)
        # # most of the achievable loss drop is buyable by learning the shrinkage
        # # coefficient alone: a smooth-looking curve with no docking skill.
        # # Do NOT annotate the value in a comment -- see _log_forced_step below,
        # # which prints the real schedule numbers at runtime.
        # idxs_step = 100
        # self._log_forced_step(idxs_step)
        # # ----------------------------------------------------------------------

        with torch.amp.autocast(device_type=device_type, enabled=False):
            return self._run_impl(prot_data_orig, idxs_step, return_time_steps)

    @staticmethod
    def _fixed_stream_seed(seed: int, stream: str) -> int:
        digest = hashlib.blake2b(
            f"{int(seed)}:{stream}".encode("utf-8"), digest_size=8
        ).digest()
        return int.from_bytes(digest, "little") & ((1 << 63) - 1)

    def run_fixed(self, prot_data_orig, idxs_step, seed, return_time_steps=False):
        """Build validation noise from deterministic, independent RNG streams."""
        device_type = prot_data_orig["cord"].device.type
        generators = {}
        for stream in ("trsl", "cdr", "seq"):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self._fixed_stream_seed(seed, stream))
            generators[stream] = generator
        generators["rota"] = self._eval_rota_sampler(
            self._fixed_stream_seed(seed, "rota")
        )
        with torch.amp.autocast(device_type=device_type, enabled=False):
            return self._run_impl(
                prot_data_orig,
                idxs_step,
                return_time_steps,
                noise_generators=generators,
            )

    def _log_forced_step(self, idxs_step):
        """AA3: print the ACTUAL schedule values once per distinct value, so that a
        stale comment can never misrepresent which regime is being trained."""
        seen = getattr(self, "_forced_step_logged", None)
        if seen is None:
            seen = set()
            self._forced_step_logged = seen
        if idxs_step in seen:
            return
        seen.add(idxs_step)
        sigma = float(self.sigmas[idxs_step])
        rms_deg = math.degrees(float(self.rota_angle_rms_schedule[idxs_step]))
        msg = (
            f"[Diffuser] FORCED idxs_step={idxs_step}  sigma={sigma:.4f}"
            f"  rota_rms={rms_deg:.2f}deg"
        )
        sd = getattr(self, "sigma_data_pose", None)
        if sd is not None:
            sd = float(sd)
            msg += (
                f"  c_skip={sd ** 2 / (sigma ** 2 + sd ** 2):.4f}"
                f"  c_out={sigma * sd / math.sqrt(sigma ** 2 + sd ** 2):.4f}"
            )
        print(msg, flush=True)

    def _run_impl(
        self,
        prot_data_orig,
        idxs_step=None,
        return_time_steps=False,
        state=None,
        noise_generators=None,
    ):
        """Build the noisy state at `idxs_step`.

        `state` (reverse sampling only) injects an externally supplied noisy state
        instead of drawing fresh noise around the ground truth.  It is a dict with
            rota_xt   : [3, 3]            noisy antibody rotation
            trsl_xt   : [3]               noisy antibody translation (PHYSICAL, i.e.
                                          the same quantity as trsl_xt_physical)
            cdr_xt    : [n_loops, lmax, n_atom, 3]  noisy CDR coords, loop-local frame
            seq_x0    : str, optional     the sequence to re-noise at this step
            generator : torch.Generator, optional  pins the sequence draw
        Everything downstream of the three draws -- the EDM coefficients, the
        local<->global assembly, the returned dict layout -- is shared verbatim
        with training, so the tensor the network sees at reverse-sampling step t
        is structurally identical to the one it saw at training step t.
        """

        if state is not None and noise_generators is not None:
            raise ValueError("state and noise_generators are mutually exclusive")

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank != self._rotation_rank:
            self.rota_sampler.generator.manual_seed(self.ROTATION_BASE_SEED + rank)
            self._rotation_rank = rank

        device = prot_data_orig["cord"].device
        dtype = torch.float32

        aa_seq_orig = prot_data_orig["seq"]
        cord_tns_orig = prot_data_orig["cords_atom14"].float()
        cmsk_mat_orig = prot_data_orig["cmsk"]
        cmsk_mat_orig14 = prot_data_orig["cmsk_atom14"]

        pmsk_vec = prot_data_orig["mask_design"]
        antibody_mask = prot_data_orig["mask_ab"].to(device=device, dtype=torch.bool)
        antigen_mask = prot_data_orig.get("antigen_mask", ~antibody_mask).to(device=device, dtype=torch.bool)
        antigen_com = cord_tns_orig[antigen_mask, 1].mean(dim=0)

        fr_mask = prot_data_orig["fr_mask"].to(device=device, dtype=torch.bool)
        cdr_mask = prot_data_orig["cdr_mask"].to(device=device, dtype=torch.bool)
        loop_masks = prot_data_orig["loop_masks"].to(device=device, dtype=torch.bool)
        loop_type_ids = prot_data_orig["loop_type_ids"].to(device=device)
        loop_left_anchor_idx = prot_data_orig["loop_left_anchor_idx"].to(device=device)
        loop_right_anchor_idx = prot_data_orig["loop_right_anchor_idx"].to(device=device)
        loop_true_len = prot_data_orig["loop_true_len"].to(device=device)
        loop_lmax = prot_data_orig["loop_lmax"].to(device=device)
        loop_occ_target = prot_data_orig["loop_occ_target"].to(device=device, dtype=torch.bool)
        loop_valid_res_mask = prot_data_orig["loop_valid_res_mask"].to(device=device, dtype=torch.bool)
        loop_atom_valid_mask = prot_data_orig["loop_atom_valid_mask"].to(device=device, dtype=torch.bool)
        loop_atom_supervise_mask = loop_valid_res_mask.unsqueeze(-1).expand_as(loop_atom_valid_mask)
        loop_global_res_indices = prot_data_orig["loop_global_res_indices"].to(device=device)
        
        # Sequence channel.  At training the x0 IS the ground-truth sequence.  In
        # reverse sampling it is the sequence fed back from the previous step, so
        # that this channel runs the same "predict x0 -> re-noise to t" loop the
        # coordinate channels do.  Without it every step would re-noise the TRUE
        # sequence, which at low t hands the answer to the network (see
        # docs/adr/0003).
        aa_seq_for_noise = (state or {}).get("seq_x0") or aa_seq_orig
        _, _, aa_seqs_pert = self._sample_probabilities(
            aa_seq_for_noise, pmsk_vec, idxs_step, device,
            generator=(state or {}).get("generator")
            if noise_generators is None else noise_generators["seq"],
        )


        # ---------------------------------------------------------
        # Modality 1: TRSL & ROTA (Antibody Rigid Translation)
        # ---------------------------------------------------------
        rota_orig, trsl_x0, ab_local_coords = self._build_antibody_rigid_params(
            cord_tns_orig,
            cmsk_mat_orig14,
            antibody_mask,
            fr_mask=fr_mask,          # X2: frame from FR atoms only
        )
        self._validate_rotation(rota_orig, "clean antibody rotation")
        
        trsl_mu = self.trsl_mu.to(device=device, dtype=dtype)
        trsl_std = self.trsl_scale.to(device=device, dtype=dtype)

        # Step 2: Physical Space Decentering
        trsl_x0_centered = trsl_x0 - trsl_mu
        
        sigma_t = self.sigmas[idxs_step].to(device=device, dtype=dtype)
        sigma_trsl = self.fr_noise_scale_trsl * sigma_t

        # Add Noise (or adopt the injected reverse-sampling state)
        if state is None:
            if noise_generators is None:
                trsl_noise = torch.randn(3, device=device, dtype=dtype)
            else:
                trsl_noise = self._randn(
                    (3,), noise_generators["trsl"], device, dtype
                )
            trsl_xt_centered = trsl_x0_centered + sigma_trsl * trsl_noise
        else:
            trsl_xt_centered = state["trsl_xt"].to(device=device, dtype=dtype).reshape(3) - trsl_mu

        # Reconstruct physical noisy coordinates for 3D Evoformer
        trsl_xt_physical = trsl_xt_centered + trsl_mu

        rota_rms = self.rota_angle_rms_schedule[idxs_step].to(device=device, dtype=dtype)
        if state is None:
            rota_sampler = (
                self.rota_sampler
                if noise_generators is None
                else noise_generators["rota"]
            )
            fr_rotation = rota_sampler.sample(
                idxs_step, device=device, dtype=torch.float32
            )
            rota_xt = torch.matmul(rota_orig.float(), fr_rotation).to(dtype=dtype)
        else:
            rota_xt = state["rota_xt"].to(device=device, dtype=dtype).reshape(3, 3)
        self._validate_rotation(rota_xt, "noisy antibody rotation")
        sigma_rota = rota_rms
        
        # Build noisy antibody complex
        noisy_ab_cord_tns = cord_tns_orig.clone()
        noisy_ab_cord_tns[antibody_mask] = local_to_global_coords(ab_local_coords, rota_xt, trsl_xt_physical)
        noisy_ab_cord_tns = noisy_ab_cord_tns * cmsk_mat_orig14.unsqueeze(-1).to(noisy_ab_cord_tns.dtype)

        # Step 3: EDM Coefficients for TRSL (sigma_data = trsl_std)
        denom_trsl = torch.sqrt(sigma_trsl**2 + trsl_std**2)
        fr_c_in   = 1.0 / denom_trsl
        fr_c_skip = (trsl_std**2) / (sigma_trsl**2 + trsl_std**2)
        fr_c_out  = (sigma_trsl * trsl_std) / denom_trsl

        # # ---------------------------------------------------------
        # # Modality 2: CDR Local Atoms
        # # ---------------------------------------------------------
        temp_coords_for_cdr_label = cord_tns_orig.clone()
        temp_coords_for_cdr_label[antibody_mask] = noisy_ab_cord_tns[    antibody_mask]
        clean_loop_local_coords, clean_anchor_rots,clean_anchor_trans = extract_per_loop_clean_local_coords(
            temp_coords_for_cdr_label,
            loop_global_res_indices,
            loop_true_len,
            loop_left_anchor_idx,
            loop_right_anchor_idx,
            loop_atom_supervise_mask,
        )

        n_loops = loop_global_res_indices.shape[0]
        n_atoms = cord_tns_orig.shape[-2]
        loop_anchor_local_coords = torch.zeros((n_loops, 2, n_atoms, 3),dtype=dtype,device=device,)
        loop_anchor_atom_mask = torch.zeros((n_loops, 2, n_atoms),dtype=torch.bool,device=device,)
        for loop_idx in range(n_loops):
            true_len = int(loop_true_len[loop_idx].item())
            left_idx = int(loop_left_anchor_idx[loop_idx].item())
            right_idx = int(loop_right_anchor_idx[loop_idx].item())

            if true_len <= 0 or left_idx < 0 or right_idx < 0:
                continue

            anchor_indices = torch.tensor(    [left_idx, right_idx],    device=device,    dtype=torch.long,)
            anchor_coords_global = temp_coords_for_cdr_label[anchor_indices]
            anchor_mask = cmsk_mat_orig[anchor_indices].to(torch.bool)

            anchor_coords_local = global_to_local_coords(
                anchor_coords_global,
                clean_anchor_rots[loop_idx],
                clean_anchor_trans[loop_idx],
            )
            anchor_coords_local = anchor_coords_local * anchor_mask.unsqueeze(-1).to(dtype)
            loop_anchor_local_coords[loop_idx] = anchor_coords_local
            loop_anchor_atom_mask[loop_idx] = anchor_mask

        cdr_mu = self.cdr_mu.to(device=device, dtype=dtype)
        cdr_std = self.cdr_scale.to(device=device, dtype=dtype)

        # Step 2: Physical Space Decentering (Masked!)
        cdr_x0_centered = clean_loop_local_coords - cdr_mu
        cdr_x0_centered = cdr_x0_centered * loop_atom_supervise_mask.unsqueeze(-1).to(dtype)

        sigma_cdr = self.cdr_local_noise_scale * sigma_t
        if state is None:
            if noise_generators is None:
                raw_local_noise = torch.randn_like(clean_loop_local_coords)
            else:
                raw_local_noise = self._randn(
                    tuple(clean_loop_local_coords.shape),
                    noise_generators["cdr"],
                    device,
                    dtype,
                )
            local_noise = sigma_cdr * raw_local_noise
            # Add Noise
            cdr_xt_centered = (cdr_x0_centered + local_noise) * loop_atom_supervise_mask.unsqueeze(-1).to(dtype)
        else:
            cdr_xt_centered = (
                state["cdr_xt"].to(device=device, dtype=dtype) - cdr_mu
            ) * loop_atom_supervise_mask.unsqueeze(-1).to(dtype)
        
        # Reconstruct physical noisy coordinates for 3D Evoformer
        noisy_loop_local_coords = (cdr_xt_centered + cdr_mu) * loop_atom_supervise_mask.unsqueeze(-1).to(dtype)

        # Step 3: EDM Coefficients for CDR (sigma_data = cdr_std)
        # Note: sigma_cdr expands if needed, but assuming scalar noise schedule per batch
        denom_cdr = torch.sqrt(sigma_cdr**2 + cdr_std**2)
        cdr_c_in   = 1.0 / denom_cdr
        cdr_c_skip = (cdr_std**2) / (sigma_cdr**2 + cdr_std**2)
        cdr_c_out  = (sigma_cdr * cdr_std) / denom_cdr

        #  noisy_ab_cord_tns as the base, for local for global
        noisy_loop_global_coords, noisy_anchor_rots, noisy_anchor_trans = rebuild_loops_from_local_coords(
            noisy_loop_local_coords,
            noisy_ab_cord_tns, 
            loop_global_res_indices,
            loop_true_len,
            loop_left_anchor_idx,
            loop_right_anchor_idx,
            loop_atom_supervise_mask,
        )

        cord_tns_noisy = merge_noisy_fr_and_loops(
            cord_tns_orig,
            noisy_ab_cord_tns,
            noisy_loop_global_coords,
            loop_global_res_indices,
            loop_true_len,
            fr_mask,
            loop_atom_supervise_mask,
        )

        # Perception mask (atom14-open, leak-free). The structural perception /
        # denoising path (cmsk_tns_init, st_encoder) must NOT see the real-atom
        # occupancy of CDR residues -- that occupancy equals n_real, a constant
        # (noise-independent) leak of the residue type. For CDR residues we use
        # the full-14 mask (cmsk_atom14 == all-True there, because build_supervision
        # fills every marker slot), which carries zero type information; non-CDR
        # residues keep their real-atom mask. cmsk-p is kept unchanged for losses.
        cmsk_perc = torch.where(
            cdr_mask.view(-1, 1), cmsk_mat_orig14.to(torch.bool), cmsk_mat_orig.to(torch.bool)
        ).to(cmsk_mat_orig.dtype)

        prot_data_pert = {
            "step": [idxs_step],
            "seq-o": aa_seq_orig,
            "cord-o": cord_tns_orig,
            "cmsk-o": cmsk_mat_orig,
            "cmsk_atom14": cmsk_mat_orig14,
            "pmsk": pmsk_vec,
            "pmsk-ligand": prot_data_orig["mask_ab"],
            # soft passthrough: only present if the datamodule supplied it;
            # consumed only by the (default-off) RAMF decodability margin loss.
            "atom14_type_target": prot_data_orig.get("atom14_type_target"),

            "seq-p": aa_seqs_pert,
            "cord-p": cord_tns_noisy.unsqueeze(0), #
            "cmsk-p": cmsk_mat_orig.unsqueeze(0),  # real-atom mask, for losses only
            "cmsk-perc": cmsk_perc.unsqueeze(0),   # atom14-open perception mask (leak-free)

            "asym-id": prot_data_orig["asym_id"].detach().clone(),
            "a-cord": prot_data_orig["a-cord"].detach().clone(),
            "a-cmsk": prot_data_orig["a-cmsk"].detach().clone(),


            "fr_mask": fr_mask.detach().clone(),
            "cdr_mask": cdr_mask.detach().clone(),
            "loop_masks": loop_masks.detach().clone(),
            "loop_type_ids": loop_type_ids.detach().clone(),
            "loop_names": list(prot_data_orig.get("loop_names", [])),
            "loop_left_anchor_idx": loop_left_anchor_idx.detach().clone(),
            "loop_right_anchor_idx": loop_right_anchor_idx.detach().clone(),
            "loop_true_len": loop_true_len.detach().clone(),
            "loop_lmax": loop_lmax.detach().clone(),
            "loop_occ_target": loop_occ_target.detach().clone(),
            "loop_valid_res_mask": loop_valid_res_mask.detach().clone(),
            "loop_atom_valid_mask": loop_atom_valid_mask.detach().clone(),
            "loop_atom_supervise_mask": loop_atom_supervise_mask.detach().clone(),
            "loop_global_res_indices": loop_global_res_indices.detach().clone(),
            "clean_loop_local_coords": clean_loop_local_coords.detach().clone(),
            "clean_coords_global": cord_tns_orig.detach().clone(),
            "noisy_loop_global_coords": noisy_loop_global_coords.detach().clone(),
            "noisy_loop_local_coords": noisy_loop_local_coords.detach().clone(),
            "anchor_frame_meta": {
                "trsl_orig": trsl_x0.detach().clone(),
                "rota_orig": rota_orig.detach().clone(),
                "rota_xt": rota_xt.detach().clone(),
                "antigen_com": antigen_com.detach().clone(),

                "trsl_xt_centered": trsl_xt_centered.detach().clone(),
                "trsl_xt_physical": trsl_xt_physical.detach().clone(),
                "trsl_mu": trsl_mu.detach().clone(),
                "trsl_scale": trsl_std.detach().clone(),

                "fr_sigma_trsl": sigma_trsl.detach().clone(),
                "fr_sigma_rota": sigma_rota.detach().clone(),
                "fr_rota_rms": rota_rms.detach().clone(),
                "fr_igso3_eps": torch.as_tensor(
                    self.rota_sampler.eps[idxs_step], device=device, dtype=dtype
                ),
                "fr_rota_is_haar": bool(idxs_step == self.n_steps),

                "fr_c_in": fr_c_in.detach().clone(),
                "fr_c_skip": fr_c_skip.detach().clone(),
                "fr_c_out": fr_c_out.detach().clone(),
            },
            "cdr_meta": {
                # Pass the centered noisy inputs for network assembly
                "cdr_xt_centered": cdr_xt_centered.detach().clone(),
                "cdr_mu": cdr_mu.detach().clone(),
                "cdr_scale": cdr_std.detach().clone(),
                
                "cdr_sigma": sigma_cdr.detach().clone(),
                
                "cdr_c_in": cdr_c_in.detach().clone(),
                "cdr_c_skip": cdr_c_skip.detach().clone(),
                "cdr_c_out": cdr_c_out.detach().clone(),
            },
            "antibody_local_coords": ab_local_coords.detach().clone(),
            "antibody_mask": antibody_mask.detach().clone(),
            "antigen_mask": antigen_mask.detach().clone(),
            "loop_anchor_local_coords": (loop_anchor_local_coords.detach().clone()),
            "loop_anchor_atom_mask": (loop_anchor_atom_mask.detach().clone()),
            "occupancy_mode": self.occupancy_mode,
            "sigama_t": {
                "cord_scale": torch.as_tensor(self.cord_scale, device=device, dtype=dtype),
                "cdr_local_noise_scale": torch.as_tensor(self.cdr_local_noise_scale, device=device, dtype=dtype),
                "sigma_raw": sigma_t.detach().clone(),
                "cdr_sigma": sigma_cdr.detach().clone(),
                "sigma_trsl":sigma_trsl.detach().clone(),
            },
        }


        if return_time_steps:
            return prot_data_pert, idxs_step
        return prot_data_pert

    # ------------------------------------------------------------------
    # Reverse sampling (evaluation only; never used on the training path)
    # ------------------------------------------------------------------

    def build_reverse_schedule(self, n_sample_steps=None):
        """Thin the full 0..T grid down to `n_sample_steps` transitions.

        Same construction as BaseDesigner._get_idxs_step so that a 200-step run
        reproduces the full grid exactly: descending, tau_0 = T, tau_K = 0.  The
        network is evaluated at every entry except the trailing 0, which only
        marks "no more noise to add".
        """
        n_full = int(self.n_steps)
        n_smpl = n_full if n_sample_steps is None else int(n_sample_steps)
        if n_smpl < 1 or n_smpl > n_full:
            raise ValueError(f"n_sample_steps must lie in [1, {n_full}], got {n_smpl}")
        alpha = n_full / n_smpl
        idxs = [int(alpha * x + 0.5) for x in range(n_smpl + 1)]
        idxs = [max(1, min(n_full, x)) for x in idxs]
        idxs[0] = 0
        idxs[-1] = n_full
        idxs.reverse()  # descending: [T, ..., 1, 0]
        # Strictly decreasing, so a coarse alpha cannot emit a duplicated t.
        dedup = [idxs[0]]
        for t in idxs[1:]:
            if t < dedup[-1]:
                dedup.append(t)
        if dedup[-1] != 0:
            dedup.append(0)
        return dedup

    def _eval_rota_sampler(self, seed):
        """IGSO(3) sampler dedicated to reverse sampling.

        A separate instance -- not `self.rota_sampler` -- because that one owns a
        PRIVATE torch.Generator that the training stream advances; drawing from it
        here would make the training noise sequence depend on how often we
        validated.  For the same reason a global torch.manual_seed cannot pin
        rotation noise: it does not reach this generator.
        """
        if self._rota_sampler_eval is None:
            self._rota_sampler_eval = OnlineIGSO3Schedule(self._rota_angle_rms_f64, seed=0)
        if seed is not None:
            self._rota_sampler_eval.generator.manual_seed(int(seed))
        return self._rota_sampler_eval

    @staticmethod
    def _randn(shape, generator, device, dtype):
        """CPU-side draw then transfer, so one CPU Generator pins every channel
        regardless of which device the run is on."""
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(
            device=device, dtype=dtype
        )

    def forward_noise_state(self, x0_state, idxs_step, generator, rota_sampler):
        """Re-noise a clean (or predicted-clean) state to `idxs_step`.

        This is deliberately the *same* algebra as the training forward pass in
        _run_impl -- x0 + sigma*eps for the two coordinate channels, right-multiply
        by an IGSO(3) draw for the rotation.  Note trsl_mu / cdr_mu cancel exactly
        (centre, add noise, un-centre), so they do not appear here.
        """
        device = x0_state["trsl_x0"].device
        dtype = torch.float32
        idxs_step = int(idxs_step)

        sigma_t = float(self.sigmas[idxs_step])
        sigma_trsl = self.fr_noise_scale_trsl * sigma_t
        sigma_cdr = self.cdr_local_noise_scale * sigma_t

        trsl_xt = x0_state["trsl_x0"].reshape(3) + sigma_trsl * self._randn(
            (3,), generator, device, dtype
        )
        cdr_x0 = x0_state["cdr_x0"]
        cdr_xt = cdr_x0 + sigma_cdr * self._randn(
            tuple(cdr_x0.shape), generator, device, dtype
        )
        fr_rotation = rota_sampler.sample(idxs_step, device=device, dtype=torch.float32)
        # Re-project onto SO(3) first: rota_x0 is the network's rota_xt @ exp(v),
        # and over a long trajectory the float32 round-off in that product drifts
        # far enough to trip _validate_rotation's 5e-4 orthogonality check.
        rota_x0 = self._project_so3(x0_state["rota_x0"].float())
        rota_xt = torch.matmul(rota_x0, fr_rotation).to(dtype)

        # seq_x0 rides along unchanged: the sequence channel's forward noising is
        # the transition matrix inside _sample_probabilities, not additive noise,
        # so it happens there rather than here.  `generator` is carried in the
        # state for the same reason -- that is where the draw takes place.
        return {
            "trsl_xt": trsl_xt,
            "rota_xt": rota_xt,
            "cdr_xt": cdr_xt,
            "seq_x0": x0_state.get("seq_x0"),
            "generator": generator,
        }

    @staticmethod
    def _project_so3(rotation):
        """Nearest rotation matrix in Frobenius norm (SVD, det forced to +1)."""
        u, _, vh = torch.linalg.svd(rotation.double())
        rot = u @ vh
        if torch.det(rot) < 0:
            u = u.clone()
            u[:, -1] = -u[:, -1]
            rot = u @ vh
        return rot.float()

    def init_reverse_state(self, prot_data_orig, generator, rota_sampler):
        """Draw the t=T starting state from the TRUE MARGINAL, not from N(mu, sigma_T).

        sigma_max/sigma_data is only ~3.07 (trsl) / ~3.33 (CDR), so the terminal
        marginal still carries 9.6% / 8.3% residual signal energy.  Starting from
        N(mu, sigma_T) would therefore be 4.9% / 4.2% too narrow -- a train/test
        mismatch at the very first and most fragile step.  sqrt(sigma_T^2 +
        sigma_data^2) is the exact marginal std.  The rotation needs no such fix:
        at t == n_steps the sampler draws exact Haar-uniform SO(3).
        """
        device = prot_data_orig["cord"].device
        dtype = torch.float32
        T = int(self.n_steps)

        sigma_t = float(self.sigmas[T])
        trsl_mu = self.trsl_mu.to(device=device, dtype=dtype)
        cdr_mu = self.cdr_mu.to(device=device, dtype=dtype)
        std_trsl = math.sqrt(
            (self.fr_noise_scale_trsl * sigma_t) ** 2 + float(self.trsl_scale) ** 2
        )
        std_cdr = math.sqrt(
            (self.cdr_local_noise_scale * sigma_t) ** 2 + float(self.cdr_scale) ** 2
        )

        n_loops, lmax = prot_data_orig["loop_global_res_indices"].shape
        n_atom = prot_data_orig["cords_atom14"].shape[-2]

        trsl_xt = trsl_mu + std_trsl * self._randn((3,), generator, device, dtype)
        cdr_xt = cdr_mu + std_cdr * self._randn(
            (n_loops, lmax, n_atom, 3), generator, device, dtype
        )
        rota_xt = rota_sampler.sample(T, device=device, dtype=torch.float32)  # Haar

        # Sequence channel: design positions start from a random residue type.
        # At t = T the transition matrix is nearly uniform so the initial letters
        # barely survive, but they are still drawn through `generator` -- anything
        # touching the global RNG would break the per-sample seed contract.
        seq_x0 = self._random_design_sequence(prot_data_orig, generator)

        return {
            "trsl_xt": trsl_xt,
            "rota_xt": rota_xt,
            "cdr_xt": cdr_xt,
            "seq_x0": seq_x0,
            "generator": generator,
        }

    @classmethod
    def _random_design_sequence(cls, prot_data_orig, generator):
        """Randomise the design positions, keep every other residue as given.

        Mirrors BaseDesigner.init_prot_data's random initialisation, except the
        draw goes through `generator` instead of the `random` module.
        """
        aa_seq = prot_data_orig["seq"]
        pmsk = prot_data_orig["mask_design"].reshape(-1).to(torch.bool).tolist()
        n_design = int(sum(pmsk))
        if n_design == 0:
            return aa_seq
        picks = torch.randint(
            len(RESD_NAMES_1C), (n_design,), generator=generator
        ).tolist()
        out, k = [], 0
        for aa, is_design in zip(aa_seq, pmsk):
            if is_design:
                out.append(RESD_NAMES_1C[picks[k]])
                k += 1
            else:
                out.append(aa)
        return ''.join(out)

    @torch.no_grad()
    def reverse_sample(
        self,
        prot_data_orig,
        forward_fn,
        n_sample_steps=None,
        seed=None,
        return_trajectory=False,
        seq_feedback=True,
    ):
        """Run the full reverse trajectory from pure noise down to t=0.

        Args:
            prot_data_orig: same dict the training path feeds to run().  Only its
                antigen / loop metadata and the FR internal conformation are used
                as conditioning; the antibody pose, the CDR conformation and (when
                seq_feedback is on) the design-position residue types are re-drawn
                from noise, so the ground truth does not leak into the trajectory.
                (The clean coords still ride along inside the dict because
                _run_impl derives the antibody's internal FR geometry and the loop
                bookkeeping from them, exactly as at training time -- this is the
                fixed-length stage's accepted `true_len` leak, nothing new.)
            forward_fn: callable(prot_data_pert) -> model outputs dict.  Supplied by
                the caller so the diffuser stays unaware of the PLM featurizer.
            n_sample_steps: number of reverse transitions; None = the full grid.
            seed: pins the whole trajectory (coordinate noise, rotations AND the
                sequence draw) for a fixed validation protocol.
            seq_feedback: close the sequence channel's loop by re-noising the
                sequence PREDICTED at the previous step instead of the ground
                truth.  False restores the old open-loop behaviour and is kept
                only as the A/B control that measures how much that leaked (see
                docs/adr/0003).

        Returns a dict with the final predicted structure and the schedule used.
        """
        device = prot_data_orig["cord"].device
        schedule = self.build_reverse_schedule(n_sample_steps)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) if seed is not None else 0)
        rota_sampler = self._eval_rota_sampler(seed if seed is not None else 0)

        state = self.init_reverse_state(prot_data_orig, generator, rota_sampler)
        if not seq_feedback:
            state["seq_x0"] = None  # falls back to prot_data_orig["seq"]

        # t=0 is the sentinel, so the lowest t the network actually sees is
        # schedule[-2]; get_temperature anneals against that, matching the way
        # AbDesigner drives its own loop.
        t_final = schedule[-2] if len(schedule) >= 2 else 0

        trajectory = []
        pred = None
        # schedule[-1] == 0 is the sentinel "nothing left to noise", so the network
        # runs on schedule[:-1] and the last of those calls produces the answer.
        for k, t in enumerate(schedule[:-1]):
            with torch.amp.autocast(device_type=device.type, enabled=False):
                prot_data_pert = self._run_impl(prot_data_orig, t, False, state=state)
            outputs = forward_fn(prot_data_pert)

            pred = {
                # [-1]: the last refinement layer is the one used downstream.
                "cord": outputs["3d"]["cord"][-1][0].detach().float(),
                "trsl_x0": outputs["3d"]["trsl"][-1][0].detach().float(),
                "rota_x0": outputs["3d"]["rota"][-1][0].detach().float(),
                # pred_x0_local lives in the anchor-local frame.  That frame is
                # rigidly attached to FR, so the local coords of a given loop
                # conformation are the same whether the anchors came from the noisy
                # or the predicted FR -- which is what makes it valid to feed this
                # straight back in as the CDR x0 for the next re-noising.
                "cdr_x0": outputs["3d"]["loop_cords"][-1][0].detach().float(),
                "step": t,
            }
            # Sequence x0 for the next step: decode directly from predicted CDR
            # geometry, not from the sequence head.  This ensures the sequence
            # channel is fully driven by the geometry channel (see ADR 0003).
            pred["seq_x0"] = self._decode_seq_from_geometry(
                pred["cdr_x0"], pred["rota_x0"], pred["trsl_x0"], prot_data_orig
            ) if seq_feedback else None
            if return_trajectory:
                trajectory.append({
                    "step": t,
                    "cord": pred["cord"].clone(),
                    "seq": pred["seq_x0"],
                })

            t_next = schedule[k + 1]
            if t_next > 0:
                state = self.forward_noise_state(
                    pred, t_next, generator, rota_sampler
                )

        if pred is None:
            raise RuntimeError("reverse schedule produced no network evaluations")

        return {
            "cord": pred["cord"],
            "trsl": pred["trsl_x0"],
            "rota": pred["rota_x0"],
            "cdr_local": pred["cdr_x0"],
            # The sequence-head chain's last sample.  Reported for comparison
            # only; the CDR types this project reports come from the geometry.
            "seq_head": pred["seq_x0"],
            "final_step": pred["step"],
            "schedule": schedule,
            "trajectory": trajectory,
        }

    def _decode_seq_from_geometry(
        self, cdr_x0_local, rota_x0, trsl_x0, prot_data_orig
    ) -> str:
        """Decode CDR sequence from predicted atom14 geometry; keep FR/Ag as-is.

        Args:
            cdr_x0_local: [n_loops, lmax, n_atom, 3] predicted CDR coords (anchor-local)
            rota_x0: [3, 3] predicted antibody rotation
            trsl_x0: [3] predicted antibody translation
            prot_data_orig: the conditioning dict WITH batch dimension [B=1, ...]

        Returns:
            Full-length sequence string with CDR positions decoded from geometry.
        """
        device = cdr_x0_local.device
        dtype = cdr_x0_local.dtype

        # Remove batch dimension where needed (prot_data_orig is batched, predictions are not)
        n_resds = len(prot_data_orig["seq"])

        # Get FR coordinates in global frame (apply FR rigid transform)
        fr_coords = prot_data_orig["cords_atom14"].clone()  # [L, 14, 3]
        ab_mask = prot_data_orig["antibody_mask"].to(torch.bool)  # [L]

        # Apply antibody rigid transform to FR positions
        fr_coords[ab_mask] = (rota_x0 @ fr_coords[ab_mask].reshape(-1, 3).T).T.reshape(-1, 14, 3) + trsl_x0

        # Reconstruct CDR in global frame using anchor-based frames
        from IgGM.utils.fr_cdr_diffusion_utils import rebuild_loops_from_local_coords

        # Extract loop metadata
        loop_global_res_indices = prot_data_orig["loop_global_res_indices"] # [n_loops, lmax]
        loop_true_len = prot_data_orig["loop_true_len"]  # [n_loops]
        loop_left_anchor_idx = prot_data_orig["loop_left_anchor_idx"]  # [n_loops]
        loop_right_anchor_idx = prot_data_orig["loop_right_anchor_idx"]  # [n_loops]
        loop_atom_valid_mask = prot_data_orig["loop_atom_valid_mask"]  # [n_loops, lmax, 14]

        cdr_global, _, _ = rebuild_loops_from_local_coords(
            coords_local=cdr_x0_local,  # [n_loops, lmax, 14, 3]
            noisy_fr_coords=fr_coords,  # [L, 14, 3] FR with rigid transform applied
            loop_global_res_indices=loop_global_res_indices,
            loop_true_len=loop_true_len,
            loop_left_anchor_idx=loop_left_anchor_idx,
            loop_right_anchor_idx=loop_right_anchor_idx,
            loop_atom_valid_mask=loop_atom_valid_mask,
        )  # [n_loops, lmax, 14, 3]

        # Build full coordinate array: start with FR, overwrite CDR positions
        cord_full = fr_coords.clone()

        for loop_idx in range(cdr_global.shape[0]):
            true_len = int(loop_true_len[loop_idx].item())
            if true_len > 0:
                global_indices = loop_global_res_indices[loop_idx, :true_len]
                cord_full[global_indices] = cdr_global[loop_idx, :true_len]

        # Build cmsk: assume all atoms valid for decoding (decoder checks actual validity)
        cmsk_full = torch.ones(n_resds, 14, device=device, dtype=torch.float32)

        # Decode: this replaces only CDR positions, FR/Ag stays as prot_data_orig["seq"]
        seq_decoded = self._atom14_sync.decode_cdr_sequence(
            seq_true=prot_data_orig["seq"],
            pred_cord_n14_tf=cord_full.cpu(),
            pred_cmsk_n14_tf=cmsk_full.cpu(),
            cdr_mask=prot_data_orig["cdr_mask"][0].cpu(),
        )
        return seq_decoded

    def __build_trmat_list(self):
        """Build a list of transition matrices."""

        # initialize basic transition matrices
        trmat_diag = torch.eye(self.n_tokns)
        trmat_unif = torch.ones((self.n_tokns, self.n_tokns)) / self.n_tokns

        # build a list of transition matrices (single step & accumulated)
        self.trmat_list_st = []  # single-step (Q_t)
        self.trmat_list_ac = []  # accumulated (\bar{Q}_t = Q_1 * Q_2 * ... * Q_t)
        for idx_step, beta in enumerate(self.seq_schedule.betas):
            if idx_step == 0:
                trmat_st = trmat_diag
                trmat_ac_prev = trmat_diag
            else:
                trmat_st = (1 - beta) * trmat_diag + beta * trmat_unif
                trmat_ac_prev = self.trmat_list_ac[-1]
            trmat_ac = torch.matmul(trmat_ac_prev, trmat_st)
            self.trmat_list_st.append(trmat_st)
            self.trmat_list_ac.append(trmat_ac)

class VarianceSchedule():
    """General variance schedule for DDPM training & sampling.

    Notes:
    > alpha_{t} = 1 - beta_{t}
    > alpha_bar_{t} = alpha_{1} * alpha_{2} * ... * alpha_{t}

    Requirements:
    > beta_{0} = 0 (which leads to alpha_{0} = 1 and alpha_bar_{0} = 1)
    > beta_{t} should be monotonically increasing
    > alpha_bar_{1} should be close to 1
    > alpha_bar_{T} should be close to 0
    """

    def __init__(self):
        """Constructor function."""

        self.n_steps = None  # integer; number of diffusion steps (T)
        self.betas = None  # 1D array of length <T+1> (from t=0 to t=T)
        self.alphas = None  # 1D array of length <T+1> (from t=0 to t=T)
        self.alphas_bar = None  # 1D array of length <T+1> (from t=0 to t=T)

    def calc_vars(self):
        """Calculate variance coefficients for forward & backward processes."""

        self.sigmas = torch.sqrt(1.0 - self.alphas_bar)
        self.betas_tld = torch.sqrt(
            self.betas[1:] * (1.0 - self.alphas_bar[:-1]) / (1.0 - self.alphas_bar[1:]))
        self.betas_tld = nn.functional.pad(self.betas_tld, (1, 0), mode='constant', value=0.0)

    def sample(self, idxs_step):
        """Build a variance schedule w/ sub-sampled time-steps to match marginal distributions."""

        assert (min(idxs_step) >= 1) and (max(idxs_step) <= self.n_steps)

        obj = VarianceSchedule()
        obj.n_steps = len(idxs_step)
        obj.alphas_bar = self.alphas_bar[[0] + sorted(idxs_step)]
        obj.alphas = torch.ones_like(obj.alphas_bar)  # t=0 corresponds to no perturbation
        obj.alphas[1:] = obj.alphas_bar[1:] / obj.alphas_bar[:-1]
        obj.betas = 1.0 - obj.alphas

        return obj

class LinearSchedule(VarianceSchedule):
    """Linear variance schedule (as proposed in DDPM)."""

    def __init__(self, n_steps=1000, beta_min=0.0001, beta_max=0.02):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_steps = n_steps
        self.beta_min = beta_min
        self.beta_max = beta_max

        # additional configurations
        self.betas = torch.linspace(self.beta_min, self.beta_max, self.n_steps)
        self.betas = nn.functional.pad(self.betas, (1, 0), mode='constant', value=0.0)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, 0)
        super().calc_vars()

class CosineSchedule(VarianceSchedule):
    """Cosine variance schedule (as proposed in Improved DDPM)."""

    def __init__(self, n_steps=4000, offset=0.008, beta_max=0.999):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_steps = n_steps
        self.offset = offset
        self.beta_max = beta_max  # to prevent singularities at the end of diffusion process

        # additional configurations
        t_vals = torch.arange(self.n_steps + 1) / self.n_steps
        f_vals = torch.cos((t_vals + offset) / (1 + offset) * np.pi / 2) ** 2
        self.betas = torch.clamp(1 - f_vals[1:] / f_vals[:-1], min=0.0, max=self.beta_max)
        self.betas = nn.functional.pad(self.betas, (1, 0), mode='constant', value=0.0)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, 0)  # re-calculated for consistency
        super().calc_vars()
