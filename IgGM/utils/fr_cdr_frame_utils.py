"""
fr_cdr_frame_utils.py
---------------------
放置路径：IgGM/utils/fr_cdr_frame_utils.py
（与现有的 fr_cdr_diffusion_utils.py 同级）

核心职责：
    提供统一的「anchor frame 对齐」工具，解决以下问题：
    - 监督信号的局部坐标（clean label）是在「当前预测 FR anchor frame」下实时重投影的，
      而不是预先在 diffuser 中计算并固定的。
    - 这样保证了 CDRLoopHead 的预测空间和 loss 的监督空间始终是同一套坐标系。
"""

from __future__ import annotations

import torch
from IgGM.utils.fr_cdr_diffusion_utils import build_anchor_frame_from_full_coords


# ---------------------------------------------------------------------------
# 1. 从全局干净坐标实时重投影到当前预测 anchor frame
# ---------------------------------------------------------------------------

def realign_clean_coords_to_pred_frame(
    clean_coords_global: torch.Tensor,          # [B, L, 14, 3] 干净全局坐标 cord_tns_orig
    loop_frame_rota: torch.Tensor,              # [B, N_loop, 3, 3] 当前预测 FR 构建的 frame
    loop_frame_trsl: torch.Tensor,              # [B, N_loop, 3]
    loop_global_res_indices: torch.Tensor,      # [B, N_loop, L_max] 每个 loop 的全局残基 idx
    loop_valid_res_mask: torch.Tensor,          # [B, N_loop, L_max] bool
    loop_atom_supervise_mask: torch.Tensor,     # [B, N_loop, L_max, 14] bool
) -> torch.Tensor:
    """
    将干净的全局原子坐标投影到「当前层预测 FR」所建立的每个 loop 局部坐标系下。

    核心逻辑：
        local = (global - trsl) @ R
    其中 R = loop_frame_rota，trsl = loop_frame_trsl，与
    _local_to_global_loop_coords 的逆变换对应。

    Args:
        clean_coords_global : [B, L_total, 14, 3]，cord_tns_orig，干净的全局坐标。
        loop_frame_rota     : [B, N_loop, 3, 3]，由当前 fr_coords anchor 构建的旋转矩阵。
        loop_frame_trsl     : [B, N_loop, 3]，由当前 fr_coords anchor 构建的平移。
        loop_global_res_indices : [B, N_loop, L_max]。
        loop_valid_res_mask : [B, N_loop, L_max]，bool。
        loop_atom_supervise_mask: [B, N_loop, L_max, 14]，bool。

    Returns:
        clean_loop_local_realigned : [B, N_loop, L_max, 14, 3]
            在当前预测 anchor frame 下的干净局部坐标（作为实时 CDR loss 的 label）。
    """
    bsz, n_loop, lmax = loop_global_res_indices.shape
    n_atom = clean_coords_global.shape[2]       # 14
    device = clean_coords_global.device
    dtype = clean_coords_global.dtype

    # step1: gather 出每个 loop 所需的干净全局坐标 [B, N_loop, L_max, 14, 3]
    idx = loop_global_res_indices.clamp_min(0)  # [B, N_loop, L_max]
    idx_expanded = idx.unsqueeze(-1).unsqueeze(-1).expand(
        bsz, n_loop, lmax, n_atom, 3
    )                                           # [B, N_loop, L_max, 14, 3]

    clean_global_expanded = clean_coords_global.unsqueeze(1).expand(
        bsz, n_loop, -1, n_atom, 3
    )                                           # [B, N_loop, L_total, 14, 3]

    clean_loop_global = torch.gather(clean_global_expanded, 2, idx_expanded)
    # [B, N_loop, L_max, 14, 3]

    # step2: 投影到每个 loop 的局部 frame：local = (global - trsl) @ R
    # loop_frame_rota: [B, N_loop, 3, 3]
    # loop_frame_trsl: [B, N_loop, 3]

    trsl = loop_frame_trsl.unsqueeze(2).unsqueeze(3)    # [B, N_loop, 1, 1, 3]
    centered = clean_loop_global - trsl                  # [B, N_loop, L_max, 14, 3]

    # R 的转置即为全局→局部的旋转：local = centered @ R  (R 为列正交阵，R^T = R^-1)
    # loop_frame_rota 已是「local→global」的 R，
    # 所以「global→local」= centered @ R （因为 R@R^T=I，local@R^T=global）
    # 验证：_local_to_global = local @ R^T + t，所以 local = (global - t) @ R
    R = loop_frame_rota                                   # [B, N_loop, 3, 3]
    # matmul: [..., 3] x [3, 3] -> [..., 3]
    R_expanded = R.unsqueeze(2).unsqueeze(3)              # [B, N_loop, 1, 1, 3, 3]
    clean_local = torch.matmul(centered.unsqueeze(-2), R_expanded).squeeze(-2)
    # [B, N_loop, L_max, 14, 3]

    # step3: mask 无效原子
    sup_mask = loop_atom_supervise_mask.to(dtype).unsqueeze(-1)  # [B, N_loop, L_max, 14, 1]
    clean_local = clean_local * sup_mask

    # step4: 无效残基清零
    valid_mask = loop_valid_res_mask.to(dtype).unsqueeze(-1).unsqueeze(-1)
    clean_local = clean_local * valid_mask

    return clean_local


# ---------------------------------------------------------------------------
# 2. 从预测 fr_coords 构建 loop anchor frames（与原 _build_loop_frames 等价，
#    但同时返回「干净坐标对齐的 label」，保证两侧坐标系一致）
# ---------------------------------------------------------------------------

def build_loop_frames_and_realign_labels(
    fr_coords: torch.Tensor,                    # [B, L, 14, 3] 当前 FRBranch 预测的全局坐标
    clean_coords_global: torch.Tensor,          # [B, L, 14, 3] 干净全局坐标 cord_tns_orig
    loop_left_anchor_idx: torch.Tensor,         # [B, N_loop]
    loop_right_anchor_idx: torch.Tensor,        # [B, N_loop]
    loop_global_res_indices: torch.Tensor,      # [B, N_loop, L_max]
    loop_valid_res_mask: torch.Tensor,          # [B, N_loop, L_max] bool
    loop_atom_supervise_mask: torch.Tensor,     # [B, N_loop, L_max, 14] bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回：
        loop_frame_rota      : [B, N_loop, 3, 3]  由 fr_coords anchor 构建的旋转矩阵
        loop_frame_trsl      : [B, N_loop, 3]      由 fr_coords anchor 构建的平移
        clean_loop_local_realigned : [B, N_loop, L_max, 14, 3]
                                     干净全局坐标在上述 frame 下的局部坐标（实时 label）
    """
    bsz = fr_coords.shape[0]
    n_loop = loop_left_anchor_idx.shape[-1]
    dtype, device = fr_coords.dtype, fr_coords.device

    frame_rota = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).repeat(bsz, n_loop, 1, 1)
    frame_trsl = torch.zeros((bsz, n_loop, 3), device=device, dtype=dtype)

    for b in range(bsz):
        coords_b = fr_coords[b]
        for i in range(n_loop):
            if not loop_valid_res_mask[b, i].any():
                continue
            left = int(loop_left_anchor_idx[b, i].item())
            right = int(loop_right_anchor_idx[b, i].item())
            if left < 0 or right < 0:
                continue
            rot, trsl = build_anchor_frame_from_full_coords(coords_b, left, right)
            frame_rota[b, i] = rot
            frame_trsl[b, i] = trsl

    # 用同一套 frame 实时重投影干净坐标，得到与预测空间一致的 label
    clean_local_realigned = realign_clean_coords_to_pred_frame(
        clean_coords_global=clean_coords_global,
        loop_frame_rota=frame_rota,
        loop_frame_trsl=frame_trsl,
        loop_global_res_indices=loop_global_res_indices,
        loop_valid_res_mask=loop_valid_res_mask,
        loop_atom_supervise_mask=loop_atom_supervise_mask,
    )

    return frame_rota, frame_trsl, clean_local_realigned
