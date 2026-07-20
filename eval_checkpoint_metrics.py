# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
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

import utils
from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.models import get_model
from qm9.sampling import sample
from qm9.analyze import check_stability
from qm9 import visualizer as vis

try:
    from qm9.rdkit_functions import BasicMolecularMetrics
    use_rdkit = True
except ModuleNotFoundError:
    use_rdkit = False


def load_args(model_path):
    with open(join(model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'
    return args


def checkpoint_name(args, epoch=None, use_ema=True):
    if epoch is None:
        if use_ema and getattr(args, 'ema_decay', 0) > 0:
            return 'generative_model_ema.npy'
        return 'generative_model.npy'

    if use_ema and getattr(args, 'ema_decay', 0) > 0:
        return f'generative_model_ema_{epoch}.npy'
    return f'generative_model_{epoch}.npy'


def append_csv(path, row):
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--checkpoint_epoch', type=int, default=None)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--batch_size_gen', type=int, default=100)
    parser.add_argument('--no_rdkit', action='store_true')
    parser.add_argument('--save_xyz', action='store_true')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--csv_path', type=str, default=None)
    parser.add_argument('--json_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    eval_args = parser.parse_args()

    assert eval_args.n_samples % eval_args.batch_size_gen == 0

    if eval_args.seed is not None:
        torch.manual_seed(eval_args.seed)

    args = load_args(eval_args.model_path)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    dtype = torch.float32
    utils.create_folders(args)

    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    generative_model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
    generative_model.to(device)

    ckpt = checkpoint_name(args, eval_args.checkpoint_epoch, use_ema=True)
    ckpt_path = join(eval_args.model_path, ckpt)
    state = torch.load(ckpt_path, map_location=device)
    generative_model.load_state_dict(state)
    generative_model.eval()

    save_dir = eval_args.save_dir
    if save_dir is None:
        tag = 'current' if eval_args.checkpoint_epoch is None else f'epoch_{eval_args.checkpoint_epoch}'
        save_dir = join(eval_args.model_path, 'eval_metrics', tag)

    count_mol_stable = 0
    count_atom_stable = 0
    count_atoms = 0
    processed = []
    start = time.time()

    print(f'Loaded checkpoint: {ckpt_path}')
    print(f'Device: {device}, n_samples: {eval_args.n_samples}, batch_size_gen: {eval_args.batch_size_gen}')

    with torch.no_grad():
        for batch_idx in range(eval_args.n_samples // eval_args.batch_size_gen):
            nodesxsample = nodes_dist.sample(eval_args.batch_size_gen)
            one_hot, charges, x, node_mask = sample(
                args, device, generative_model, dataset_info,
                prop_dist=prop_dist, nodesxsample=nodesxsample)

            atomsxmol = torch.sum(node_mask, dim=1).long().cpu()
            for i in range(eval_args.batch_size_gen):
                n = int(atomsxmol[i])
                atom_type = one_hot[i, :n].argmax(1).cpu().detach()
                pos = x[i, :n].cpu().detach()
                mol_stable, stable_bonds, natoms = check_stability(pos, atom_type, dataset_info)
                count_mol_stable += int(mol_stable)
                count_atom_stable += int(stable_bonds)
                count_atoms += int(natoms)
                processed.append((pos, atom_type))

            if eval_args.save_xyz:
                vis.save_xyz_file(
                    save_dir, one_hot, charges, x, dataset_info,
                    id_from=batch_idx * eval_args.batch_size_gen,
                    name='molecule', node_mask=node_mask)

            done = (batch_idx + 1) * eval_args.batch_size_gen
            print(f'{done}/{eval_args.n_samples} mol_stable={count_mol_stable / done:.4f} '
                  f'atom_stable={count_atom_stable / count_atoms:.4f} '
                  f'sec/sample={(time.time() - start) / done:.3f}', flush=True)

    metrics = {
        'model_path': eval_args.model_path,
        'checkpoint': ckpt,
        'checkpoint_epoch': eval_args.checkpoint_epoch,
        'n_samples': eval_args.n_samples,
        'mol_stable': count_mol_stable / float(eval_args.n_samples),
        'atom_stable': count_atom_stable / float(count_atoms),
        'seconds': time.time() - start,
    }

    if use_rdkit and not eval_args.no_rdkit:
        rdkit_metrics, unique = BasicMolecularMetrics(dataset_info).evaluate(processed)
        validity, uniqueness, novelty = rdkit_metrics
        metrics.update({
            'validity': validity,
            'uniqueness_given_valid': uniqueness,
            'novelty_given_unique': novelty,
            'valid_unique': validity * uniqueness,
            'n_unique_valid': len(unique) if unique is not None else 0,
        })
    else:
        print('Skipping RDKit metrics.')

    print(json.dumps(metrics, indent=2, sort_keys=True))

    if eval_args.csv_path is not None:
        append_csv(eval_args.csv_path, metrics)
        print(f'Wrote CSV row: {eval_args.csv_path}')

    if eval_args.json_path is not None:
        with open(eval_args.json_path, 'w') as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        print(f'Wrote JSON: {eval_args.json_path}')


if __name__ == '__main__':
    main()
