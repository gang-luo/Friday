# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.distributions import Categorical

from IgGM.model import build_iggm_modules
from IgGM.protein import ProtStruct, build_antibody_region_metadata
from IgGM.protein.data_transform import get_asym_ids
from IgGM.utils import (
    IGSO3Buffer,
    merge_noisy_fr_and_loops,
    rebuild_loops_from_local_coords,
    replace_with_mask,
    to_device,
)
from .base_designer import BaseDesigner
from ..model.arch.core.diffuser import Diffuser


class AbDesigner(BaseDesigner):
    """The antibody & antigen multimer structure predictor."""

    def __init__(self, ppi_path, design_path, buffer_path, config):
        super().__init__()
        logging.info('restoring the pre-trained IgGM-PPI-SeqPT model ...')
        self.plm_featurizer, self.model, c_s, c_p = build_iggm_modules(
            ppi_path=ppi_path,
            design_path=design_path,
            config=config,
        )
        config.c_s = c_s if c_s is not None else getattr(config, 'c_s', None)
        config.c_p = c_p if c_p is not None else getattr(config, 'c_p', None)
        self.config = config
        self.diffusion_mode = getattr(config, 'diffusion_mode', 'legacy')
        self.structure_mode = getattr(config, 'structure_mode', self.diffusion_mode)
        self.loss_mode = getattr(config, 'loss_mode', 'legacy')
        self.occupancy_prediction_mode = getattr(
            config,
            'occupancy_prediction_mode',
            getattr(config, 'occupancy_mode', 'joint_predict'),
        )
        self.occupancy_threshold = float(getattr(config, 'occupancy_threshold', 0.5))

        self.igso3_buffer = IGSO3Buffer()
        self.igso3_buffer.load(buffer_path)
        self.diffuser = Diffuser(
            igso3_buffer=self.igso3_buffer,
            diffusion_mode=self.diffusion_mode,
            fr_noise_scale_trsl=getattr(config, 'fr_noise_scale_trsl', 1.0),
            fr_noise_scale_rota=getattr(config, 'fr_noise_scale_rota', 1.0),
            cdr_local_noise_scale=getattr(config, 'cdr_local_noise_scale', 1.0),
            occupancy_mode=self.occupancy_prediction_mode,
        )
        self.buffer_path = buffer_path
        logging.info('restoring the pre-trained IgGM design model ... done')
        self.idxs_step = self._get_idxs_step()
        self.eval()

    @staticmethod
    def _find_design_spans(seq: str) -> List[Tuple[int, int]]:
        spans = []
        start = None
        for idx, aa in enumerate(seq):
            if aa == 'X' and start is None:
                start = idx
            elif aa != 'X' and start is not None:
                spans.append((start, idx - 1))
                start = None
        if start is not None:
            spans.append((start, len(seq) - 1))
        return spans

    def _infer_cdr_sequences(self, chains: List[Dict[str, object]]) -> Dict[str, List[int]]:
        cdr = {f'cdr_{name}': [] for name in ('H1', 'H2', 'H3', 'L1', 'L2', 'L3')}
        if not chains:
            return cdr
        heavy_spans = self._find_design_spans(chains[0]['sequence']) if len(chains) >= 1 else []
        light_spans = self._find_design_spans(chains[1]['sequence']) if len(chains) == 3 else []
        for idx, loop_name in enumerate(('H1', 'H2', 'H3')):
            if idx < len(heavy_spans):
                stt, end = heavy_spans[idx]
                cdr[f'cdr_{loop_name}'] = list(range(stt + 1, end + 2))
        for idx, loop_name in enumerate(('L1', 'L2', 'L3')):
            if idx < len(light_spans):
                stt, end = light_spans[idx]
                cdr[f'cdr_{loop_name}'] = list(range(stt + 1, end + 2))
        return cdr

    def _attach_region_metadata(self, inputs, chains, complex_id):
        if self.diffusion_mode != 'fr_cdr_sync' and self.structure_mode != 'fr_cdr_sync':
            return None
        complex_data = inputs[complex_id]
        sequence_lengths = {
            'H': len(chains[0]['sequence']) if len(chains) >= 1 else 0,
            'L': len(chains[1]['sequence']) if len(chains) == 3 else 0,
            'A': len(chains[-1]['sequence']),
        }
        cdr_sequences = self._infer_cdr_sequences(chains)
        metadata = build_antibody_region_metadata(
            sequence_lengths=sequence_lengths,
            cdr_sequences=cdr_sequences,
            atom_mask=complex_data['base']['cmsk'],
        )
        complex_data.update(metadata)
        complex_data['loop_names'] = list(metadata['loop_names'])
        complex_data['cdr_sequences'] = cdr_sequences
        complex_data['sequence_lengths'] = sequence_lengths
        if not any(len(v) > 0 for v in cdr_sequences.values()):
            logging.warning('fr_cdr_sync requested but no loop spans were inferred from X-masks; loop-specific updates will degenerate to empty-loop behavior.')
        return metadata

    def _build_inputs(self, chains, task='design'):
        num_chains = len(chains)
        inputs = super()._build_inputs(chains, task)
        chain_ids = inputs['base']['chain_ids']
        if num_chains == 2:
            ligand_id = chain_ids[0]
        else:
            ligand_id = ':'.join(chain_ids[:2])
            h_seq = inputs[chain_ids[0]]['base']['seq']
            l_seq = inputs[chain_ids[1]]['base']['seq']
            inputs[ligand_id] = {
                'base': {'seq': h_seq + l_seq},
                'asym_id': get_asym_ids([h_seq, l_seq]).unsqueeze(dim=0),
                'feat': {},
            }

        inputs['base']['ligand_id'] = ligand_id
        inputs['base']['receptor_id'] = chain_ids[-1]

        complex_id = ':'.join([ligand_id, chain_ids[-1]])
        prot_data = self.init_prot_data(inputs, complex_id)
        prot_data['base']['complex_id'] = complex_id
        epitope = prot_data[complex_id]['epitope']
        if epitope is None:
            logging.info('no epitope information provided, the position placement will be determined by the model')
        self._attach_region_metadata(prot_data, chains, complex_id)
        prot_data[complex_id]['diffusion_mode'] = self.diffusion_mode
        prot_data[complex_id]['structure_mode'] = self.structure_mode
        prot_data[complex_id]['loss_mode'] = self.loss_mode
        return prot_data

    def forward(self, inputs, chunk_size=None, temperature=1.0):
        start = time.time()
        inputs = to_device(inputs, device=self.device)
        complex_id = inputs['base']['complex_id']
        idxs_step = self.idxs_step[:-1]
        prot_data_curr = inputs[complex_id]
        inputs_addi = None
        for idx_step in idxs_step:
            aa_seqs_pred, cord_tns_pred, cmsk_tns_pred, inputs_addi, aux_meta = self.__sample_cm_ss2ss(
                prot_data_curr,
                idx_step,
                inputs_addi,
                chunk_size=chunk_size,
                temperature=temperature,
            )
            prot_data_curr['seq'] = aa_seqs_pred
            prot_data_curr['cord'] = cord_tns_pred
            prot_data_curr['cmsk'] = cmsk_tns_pred
            if aux_meta is not None:
                prot_data_curr.update(aux_meta)
        logging.info('start ab design model in %.2f second', time.time() - start)
        return prot_data_curr

    @torch.no_grad()
    def infer(self, chains, task='design', *args, **kwargs):
        assert len(chains) in (2, 3), f'FASTA file should contain 2 or 3 chains'
        inputs = self._build_inputs(chains, task=task)
        outputs = self.forward(inputs, *args, **kwargs)
        return inputs, outputs

    def infer_pdb(self, chains, filename, relax=False, task='design', *args, **kwargs):
        inputs, outputs = self.infer(chains, task, *args, **kwargs)
        self._output_to_fasta(inputs, outputs, filename[:-4] + '.fasta')
        if task == 'design' or task == 'fr_design':
            self._output_to_pdb(inputs, outputs, filename, relax=relax)

    def _predict_loop_lengths(self, occ_logits, loop_valid_res_mask, loop_true_len):
        probs = torch.sigmoid(occ_logits)
        pred_lens = torch.zeros_like(loop_true_len)
        for idx in range(loop_true_len.shape[0]):
            valid = loop_valid_res_mask[idx].to(torch.bool)
            valid_len = int(valid.sum().item())
            if valid_len == 0:
                pred_lens[idx] = 0
                continue
            if self.occupancy_prediction_mode in {'argmax_prefix', 'joint_predict', 'prefix_threshold'}:
                flags = probs[idx, :valid_len] >= self.occupancy_threshold
                prefix_len = 0
                for flag in flags.tolist():
                    if flag:
                        prefix_len += 1
                    else:
                        break
                pred_lens[idx] = prefix_len
            else:
                pred_lens[idx] = min(valid_len, int(loop_true_len[idx].item()))
        return pred_lens

    def _assemble_fr_cdr_coords(self, inputs, outputs, aa_seqs_pred):
        bundle = outputs.get('3d', {}).get('fr_cdr')
        if bundle is None:
            return None
        pred_local = bundle['cdr']['pred_local_coords'][0]
        occ_logits = bundle['cdr']['pred_occupancy_logits'][0]
        fr_pred_coords = bundle['fr']['pred_coords'][0]
        merged_coords = bundle.get('merged', {}).get('coords', None)
        loop_valid_res_mask = inputs['loop_valid_res_mask'].to(torch.bool)
        loop_true_len = inputs['loop_true_len'].to(torch.long)
        pred_loop_len = self._predict_loop_lengths(occ_logits, loop_valid_res_mask, loop_true_len)
        loop_atom_valid_mask = inputs['loop_atom_valid_mask'].to(torch.bool).clone()
        for idx in range(loop_atom_valid_mask.shape[0]):
            loop_atom_valid_mask[idx, pred_loop_len[idx]:] = False
        loop_global_res_indices = inputs['loop_global_res_indices'].to(torch.long)
        loop_left_anchor_idx = inputs['loop_left_anchor_idx'].to(torch.long)
        loop_right_anchor_idx = inputs['loop_right_anchor_idx'].to(torch.long)
        if merged_coords is not None:
            full_coords = merged_coords[0]
        else:
            loop_global_coords, _, _ = rebuild_loops_from_local_coords(
                pred_local,
                fr_pred_coords,
                loop_global_res_indices,
                pred_loop_len,
                loop_left_anchor_idx,
                loop_right_anchor_idx,
                loop_atom_valid_mask,
            )
            full_coords = merge_noisy_fr_and_loops(
                inputs['cord-p'][0],
                fr_pred_coords,
                loop_global_coords,
                loop_global_res_indices,
                pred_loop_len,
                inputs['fr_mask'],
                loop_atom_valid_mask,
            )
        pmsk_vec_ligand = inputs['pmsk-ligand'].to(torch.bool)
        full_coords = torch.where(pmsk_vec_ligand.view(-1, 1, 1), full_coords, inputs['cord-o'])
        full_cmsk = inputs['cmsk-p'][0].clone()
        loop_export_mask = torch.ones(full_cmsk.shape[0], dtype=torch.bool, device=full_cmsk.device)
        next_loop_valid_res_mask = inputs['loop_valid_res_mask'].to(torch.bool).clone()
        next_loop_occ_target = torch.zeros_like(next_loop_valid_res_mask)
        for idx in range(loop_global_res_indices.shape[0]):
            keep_len = int(pred_loop_len[idx].item())
            next_loop_valid_res_mask[idx] = False
            next_loop_occ_target[idx] = False
            if keep_len > 0:
                next_loop_valid_res_mask[idx, :keep_len] = True
                next_loop_occ_target[idx, :keep_len] = True
            true_len = int(loop_true_len[idx].item())
            if true_len <= keep_len:
                continue
            drop_idx = loop_global_res_indices[idx, keep_len:true_len]
            drop_idx = drop_idx[drop_idx >= 0]
            if drop_idx.numel() == 0:
                continue
            full_cmsk[drop_idx] = 0
            loop_export_mask[drop_idx] = False
        full_cmsk = torch.where(pmsk_vec_ligand.view(-1, 1), full_cmsk, inputs['cmsk-o'])
        return {
            'seq': aa_seqs_pred[0],
            'cord': full_coords,
            'cmsk': full_cmsk,
            'loop_export_mask': loop_export_mask,
            'pred_loop_len': pred_loop_len.detach().clone(),
            'occupancy_logits': occ_logits.detach().clone(),
            'loop_true_len': pred_loop_len.detach().clone(),
            'loop_valid_res_mask': next_loop_valid_res_mask.detach().clone(),
            'loop_occ_target': next_loop_occ_target.detach().clone(),
            'loop_atom_valid_mask': loop_atom_valid_mask.detach().clone(),
        }

    def __sample_cm_ss2ss(self, prot_data_curr, idx_step, inputs_addi, chunk_size=None, temperature=1.0):
        inputs = self.__build_inputs_cm(prot_data_curr, idx_step)
        outputs = self.model(inputs, inputs_addi=inputs_addi, chunk_size=chunk_size)
        inputs_addi = self.build_inputs_addi(outputs)
        temperature = get_temperature(idx_step, self.idxs_step[-1])
        prob_tns = nn.functional.softmax(outputs['1d'].permute(0, 2, 1) / temperature, dim=2)
        distr = Categorical(probs=prob_tns)
        aa_seqs_pred = self.sample_seqs_from_distr(distr)
        pmsk_vec = inputs['pmsk']
        aa_seqs_pred = [replace_with_mask(inputs['seq-o'], aa_seq_pred, pmsk_vec) for aa_seq_pred in aa_seqs_pred]

        aux_meta = None
        if self.diffusion_mode == 'fr_cdr_sync' and outputs.get('3d', {}).get('fr_cdr') is not None:
            aux_meta = self._assemble_fr_cdr_coords(inputs, outputs, aa_seqs_pred)
            if aux_meta is not None:
                cord_tns_pred = aux_meta['cord'].unsqueeze(0)
                cmsk_tns_pred = aux_meta['cmsk'].unsqueeze(0)
                inputs_addi['cord'] = cord_tns_pred.detach().clone()
                next_meta = {k: v for k, v in aux_meta.items() if k not in {'seq', 'cord', 'cmsk'}}
                return aa_seqs_pred[0], cord_tns_pred[0], cmsk_tns_pred[0], inputs_addi, next_meta

        cord_tns_pred = self.calc_cords_from_param(aa_seqs_pred, outputs['3d']['param'][-1])
        pmsk_vec_ligand = inputs['pmsk-ligand']
        cord_tns_pred = torch.where(pmsk_vec_ligand.view(1, -1, 1, 1).to(torch.bool), cord_tns_pred, inputs['cord-o'])
        cmsk_tns_pred = ProtStruct.get_cmsk_vld(aa_seqs_pred[0], self.device).view(1, -1, inputs['cmsk-o'].shape[-1])
        cmsk_tns_pred = torch.where(pmsk_vec_ligand.view(1, -1, 1).to(torch.bool), cmsk_tns_pred, inputs['cmsk-o'])
        return aa_seqs_pred[0], cord_tns_pred[0], cmsk_tns_pred[0], inputs_addi, None

    def __build_inputs_cm(self, prot_data_curr, idx_step):
        prot_data_pert = self.diffuser.run(prot_data_curr, idx_step)
        inputs = self.model.featurize(self.plm_featurizer, prot_data_pert)
        inputs['structure_mode'] = self.structure_mode
        if prot_data_curr['contact'] is None:
            ic_feat = torch.zeros_like(prot_data_curr['asym_id'])
            ag_len = len(prot_data_curr['epitope'])
            ic_feat[:, -ag_len:] = prot_data_curr['epitope']
            inputs['ic_feat'] = ic_feat.unsqueeze(-1).type_as(inputs['sfea-i'])
        else:
            bs, length = prot_data_curr['asym_id'].shape
            ic_feat = torch.zeros(bs, length, length).to(prot_data_curr['asym_id'].device)
            ic_feat[:, ...] = prot_data_curr['contact']
            inputs['ic_feat'] = ic_feat.unsqueeze(-1).type_as(inputs['sfea-i'])
        return inputs


def get_temperature(t, total_steps):
    base_T = 2
    min_T = 0.5
    return base_T - (base_T - min_T) * (1 - t / 200)
