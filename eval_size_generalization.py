try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ModuleNotFoundError:
    pass

import argparse
import csv
import json
import os
import pickle
import time
from os.path import join

import torch
from ase import Atoms
from ase.io import write as ase_write

from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.analyze import check_stability
from qm9.models import get_model
from qm9.rdkit_functions import BasicMolecularMetrics


def load_args(model_path):
    with open(join(model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'
    return args


def sample_fixed_n(args, model, device, dataset_info, n_nodes, batch_size):
    node_mask = torch.ones(batch_size, n_nodes, 1, device=device)
    edge_mask = (1 - torch.eye(n_nodes, device=device)).unsqueeze(0)
    edge_mask = edge_mask.repeat(batch_size, 1, 1).view(batch_size * n_nodes * n_nodes, 1)
    context = None
    x, h = model.sample(batch_size, n_nodes, node_mask, edge_mask, context, fix_noise=False)
    return h['categorical'], h['integer'], x, node_mask


def to_ase_atoms(pos, atom_type, dataset_info):
    symbols = [dataset_info['atom_decoder'][int(atom)] for atom in atom_type.tolist()]
    return Atoms(symbols=symbols, positions=pos.numpy())


def evaluate_size(args, model, device, dataset_info, metrics, n_nodes, n_samples, batch_size, save_dir=None):
    assert n_samples % batch_size == 0
    processed = []
    ase_frames = []
    n_mol_stable = 0
    n_atom_stable = 0
    n_atom_total = 0
    n_heavy = 0
    atom_counts = {symbol: 0 for symbol in dataset_info['atom_decoder']}

    with torch.no_grad():
        for _ in range(n_samples // batch_size):
            one_hot, charges, x, node_mask = sample_fixed_n(
                args, model, device, dataset_info, n_nodes, batch_size)
            for i in range(batch_size):
                atom_type = one_hot[i].argmax(1).detach().cpu()
                pos = x[i].detach().cpu()
                mol_stable, atom_stable, atom_total = check_stability(pos, atom_type, dataset_info)
                n_mol_stable += int(mol_stable)
                n_atom_stable += int(atom_stable)
                n_atom_total += int(atom_total)
                for atom in atom_type.tolist():
                    symbol = dataset_info['atom_decoder'][atom]
                    atom_counts[symbol] += 1
                    if symbol != 'H':
                        n_heavy += 1
                processed.append((pos, atom_type))
                if save_dir is not None:
                    ase_frames.append(to_ase_atoms(pos, atom_type, dataset_info))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        ase_write(join(save_dir, f'n_atoms_{n_nodes:02d}.extxyz'), ase_frames, format='extxyz')

    valid, validity = metrics.compute_validity(processed)
    unique_valid, uniqueness = metrics.compute_uniqueness(valid) if len(valid) > 0 else ([], 0.0)

    return {
        'n_atoms': n_nodes,
        'n_samples': n_samples,
        'mean_heavy_atoms': n_heavy / n_samples,
        'mol_stable': n_mol_stable / n_samples,
        'atom_stable': n_atom_stable / n_atom_total,
        'validity': validity,
        'uniqueness_given_valid': uniqueness,
        'valid_unique': validity * uniqueness,
        **{f'frac_{symbol}': atom_counts[symbol] / n_atom_total for symbol in atom_counts},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='outputs/edm_qm9')
    parser.add_argument('--n_min', type=int, default=21)
    parser.add_argument('--n_max', type=int, default=45)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--csv_path', default=None)
    parser.add_argument('--json_path', default=None)
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--seed', type=int, default=None)
    args_eval = parser.parse_args()

    if args_eval.seed is not None:
        torch.manual_seed(args_eval.seed)

    args = load_args(args_eval.model_path)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
    model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
    model.to(device)
    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    model.load_state_dict(torch.load(join(args_eval.model_path, fn), map_location=device))
    model.eval()

    metrics = BasicMolecularMetrics(dataset_info)
    rows = []
    start = time.time()
    for n_nodes in range(args_eval.n_min, args_eval.n_max + 1):
        row = evaluate_size(args, model, device, dataset_info, metrics,
                            n_nodes, args_eval.n_samples, args_eval.batch_size,
                            save_dir=args_eval.save_dir)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    if args_eval.csv_path is not None:
        os.makedirs(os.path.dirname(args_eval.csv_path), exist_ok=True)
        with open(args_eval.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f'Wrote CSV: {args_eval.csv_path}')

    if args_eval.json_path is not None:
        os.makedirs(os.path.dirname(args_eval.json_path), exist_ok=True)
        with open(args_eval.json_path, 'w') as f:
            json.dump(rows, f, indent=2)
        print(f'Wrote JSON: {args_eval.json_path}')

    print(f'Total seconds: {time.time() - start:.1f}')


if __name__ == '__main__':
    main()
