# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
from .comm import all_logging_disabled, get_rand_str
from .tensor import cdist, clone, to_device, to_tensor
from .registry import Registry
from .file import jload, jdump, get_tmp_dpath, download_file
from .env import seed_all_rng, setup_logger, setup
from .diff_util import ss2ptr, ptr2ss, prob2seq, so3_scale, intp_prob_mat_dsct, intp_trsl_mat, intp_rota_tns, IsotropicGaussianSO3, OnlineIGSO3Schedule, IGSO3Buffer, rota2quat, replace_with_mask, calc_trsl_vec,skew2vec,log_rmat,so3_log_vector
from .fr_cdr_diffusion_utils import (
    apply_rigid_transform_coords,
    apply_rigid_transform_to_masked_coords,
    build_anchor_frame_from_full_coords,
    extract_clean_fr_reference,
    extract_per_loop_clean_local_coords,
    global_to_local_coords,
    local_to_global_coords,
    merge_noisy_fr_and_loops,
    rebuild_loops_from_local_coords,
    rebuild_and_merge_loops,
)

# Relax the input pdb file
def Rosetta_relax(pdb_file):
    import os
    import pyrosetta
    pyrosetta.init()
    from pyrosetta.rosetta.core.select import residue_selector as selections
    from pyrosetta import pose_from_pdb, create_score_function
    from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
    from pyrosetta.rosetta.core.select.movemap import MoveMapFactory, move_map_action
    from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
    from pyrosetta.rosetta.protocols.relax import FastRelax

    print(f'Rosetta processing {pdb_file} for Relax')

    pose = pose_from_pdb(pdb_file)
    scorefxn = create_score_function('ref2015')

    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.RestrictToRepacking())
    tf.push_back(operation.PreventRepacking())
    flexible_dict = dict()
    count = 0

    all_residue_selector = selections.ResidueIndexSelector()
    if pose.num_chains() == 2:
        all_residue_selector.set_index_range(1, len(pose.chain_sequence(1)))
    else:
        all_residue_selector.set_index_range(1, len(pose.chain_sequence(1)) + len(
            pose.chain_sequence(2)))


    nbr_selector = selections.NeighborhoodResidueSelector()
    nbr_selector.set_focus_selector(all_residue_selector)
    nbr_selector.set_include_focus_in_subset(True)
    subset_selector = nbr_selector
    prevent_repacking_rlt = operation.PreventRepackingRLT()
    prevent_subset_repacking = operation.OperateOnResidueSubset(
        prevent_repacking_rlt,
        subset_selector,
        flip_subset=True,
    )
    tf.push_back(prevent_subset_repacking)
    packer_task = tf.create_task_and_apply_taskoperations(pose)

    movemap = MoveMapFactory()
    movemap.add_bb_action(move_map_action.mm_enable, all_residue_selector)
    movemap.add_chi_action(move_map_action.mm_enable, subset_selector)
    mm = movemap.create_movemap_from_pose(pose)
    fastrelax = FastRelax()
    fastrelax.set_scorefxn(scorefxn)
    fastrelax.set_movemap(mm)
    fastrelax.set_task_factory(tf)
    fastrelax.apply(pose)

    pose.dump_pdb(f'{pdb_file}')
