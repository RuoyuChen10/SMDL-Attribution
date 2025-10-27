import os
import argparse
import json
import subprocess
from typing import List, Tuple

import numpy as np
import cv2
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def list_pairs(explanation_dir: str) -> List[Tuple[str, str, str]]:
    npy_root = os.path.join(explanation_dir, 'npy')
    json_root = os.path.join(explanation_dir, 'json')
    pairs = []
    if not os.path.isdir(npy_root) or not os.path.isdir(json_root):
        return pairs
    for cls in sorted(os.listdir(npy_root)):
        npy_cls = os.path.join(npy_root, cls)
        json_cls = os.path.join(json_root, cls)
        if not os.path.isdir(npy_cls) or not os.path.isdir(json_cls):
            continue
        for fname in sorted(os.listdir(npy_cls)):
            if not fname.endswith('.npy'):
                continue
            npy_path = os.path.join(npy_cls, fname)
            json_path = os.path.join(json_cls, fname.replace('.npy', '.json'))
            if os.path.isfile(json_path):
                pairs.append((cls, npy_path, json_path))
    return pairs


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_overlay(npy_array: np.ndarray, out_path: str):
    overlay = np.clip(npy_array.sum(0), 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, overlay[..., ::-1])


def save_step_gif(npy_array: np.ndarray, out_path: str, duration: float = 0.2, max_steps: int = 0):
    frames = []
    cum = np.zeros_like(npy_array[0])
    steps = npy_array.shape[0] if max_steps in (0, None) else min(max_steps, npy_array.shape[0])
    for i in range(steps):
        cum = np.clip(cum + npy_array[i], 0, 255).astype(np.uint8)
        frames.append(cum[..., ::-1])  # BGR->RGB for gif
    imageio.mimsave(out_path, frames, duration=duration)


essentials = ['consistency_score', 'collaboration_score', 'effectiveness_score', 'smdl_score']

def save_score_plot(json_data: dict, out_path: str):
    plt.figure(figsize=(8, 4))
    for k in essentials:
        if k in json_data and isinstance(json_data[k], list) and len(json_data[k]) > 0:
            plt.plot(np.arange(1, len(json_data[k]) + 1), json_data[k], label=k)
    plt.xlabel('step')
    plt.ylabel('score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def write_summary(csv_path: str, rows: List[dict]):
    import csv
    if not rows:
        return
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_auc_eval(explanation_dir: str, out_txt: str):
    ensure_dir(os.path.dirname(out_txt))
    cmd = [
        'python', '-m', 'evals.eval_AUC_faithfulness',
        '--explanation-dir', explanation_dir
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        with open(out_txt, 'w', encoding='utf-8') as f:
            f.write(res.stdout)
            if res.stderr:
                f.write('\n[stderr]\n')
                f.write(res.stderr)
    except subprocess.CalledProcessError as e:
        with open(out_txt, 'w', encoding='utf-8') as f:
            f.write('[AUC script failed]\n')
            f.write(e.stdout or '')
            f.write('\n[stderr]\n')
            f.write(e.stderr or '')


def main():
    parser = argparse.ArgumentParser(description='Postprocess submodular results: overlays, gifs, score plots, summary, AUC eval')
    parser.add_argument('--explanation-dir', type=str, required=True, help='Path like submodular_results/.../slico-...')
    parser.add_argument('--out-dir', type=str, default='', help='Output root (default: <explanation-dir>/postprocess)')
    parser.add_argument('--gif', action='store_true', help='Export step GIF')
    parser.add_argument('--gif-steps', type=int, default=0, help='Max steps in GIF (0 = all)')
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.explanation_dir)
    out_root = os.path.abspath(args.out_dir or os.path.join(exp_dir, 'postprocess'))

    pairs = list_pairs(exp_dir)
    if not pairs:
        print('No pairs found under:', exp_dir)
        return

    # Export per-sample artifacts
    rows = []
    for cls, npy_path, json_path in pairs:
        rel_name = os.path.splitext(os.path.basename(npy_path))[0]
        sample_out = os.path.join(out_root, cls, rel_name)
        ensure_dir(sample_out)

        # Load
        arr = np.load(npy_path)  # [steps,H,W,3]
        with open(json_path, 'r', encoding='utf-8') as f:
            info = json.load(f)

        # overlay
        save_overlay(arr, os.path.join(sample_out, 'overlay.jpg'))

        # gif (optional)
        if args.gif:
            save_step_gif(arr, os.path.join(sample_out, 'step.gif'), duration=0.2, max_steps=args.gif_steps)

        # score plot
        save_score_plot(info, os.path.join(sample_out, 'scores.png'))

        # summary row
        rows.append({
            'class_id': cls,
            'image': rel_name,
            'steps': int(arr.shape[0]),
            'final_consistency': (info.get('consistency_score') or [None])[-1],
            'final_collaboration': (info.get('collaboration_score') or [None])[-1],
            'smdl_score_max': info.get('smdl_score_max'),
            'smdl_score_max_index': info.get('smdl_score_max_index'),
        })

    # Write summary
    write_summary(os.path.join(out_root, 'summary.csv'), rows)

    # Run AUC eval and dump output
    run_auc_eval(exp_dir, os.path.join(out_root, 'auc.txt'))

    print('Done. Outputs saved under:', out_root)


if __name__ == '__main__':
    main() 