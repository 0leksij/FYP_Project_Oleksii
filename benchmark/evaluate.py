#!/usr/bin/env python3
"""
Main evaluation pipeline for text-to-3D meshes
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm

from metrics.mesh_metrics import safe_compute_metrics
from metrics.clip_metrics import CLIPEvaluator, render_multiview_images
from metrics.image_reward_metrics import ImageRewardEvaluator


_RENDER_ZERO = {
    'clip_score_mean':       0.0,
    'clip_score_std':        0.0,
    'clip_score_min':        0.0,
    'clip_score_max':        0.0,
    'multiview_consistency': 0.0,
    'image_reward_mean':     0.0,
    'image_reward_std':      0.0,
    'image_reward_min':      0.0,
    'image_reward_max':      0.0,
}


class MeshEvaluationPipeline:
    """Complete evaluation pipeline for text-to-3D generated meshes"""

    def __init__(self,
                 output_dir: str = "results",
                 render_dir: str = "renders",
                 verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.render_dir = Path(render_dir)
        self.verbose = verbose

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.render_dir.mkdir(parents=True, exist_ok=True)

        self.clip_evaluator         = CLIPEvaluator()
        self.image_reward_evaluator = ImageRewardEvaluator()

    def evaluate_single_mesh(self,
                             mesh_path: str,
                             prompt: str,
                             mesh_id: str,
                             reference_path: Optional[str] = None,
                             skip_rendering: bool = False) -> Dict:
        results = {
            'mesh_id':        mesh_id,
            'mesh_path':      mesh_path,
            'prompt':         prompt,
            'reference_path': reference_path if reference_path else 'none',
        }

        if self.verbose:
            print(f"\nEvaluating: {mesh_id}")
            print(f"Prompt: {prompt}")

        # 1. Mesh geometry + topology + regularity metrics
        if self.verbose:
            print("  Computing mesh metrics...")
        results.update(safe_compute_metrics(
            mesh_path,
            reference_path=reference_path,
            verbose=self.verbose,
        ))

        # 2. Render → CLIP / consistency / ImageReward
        if skip_rendering or results.get('load_success', 0) == 0:
            results.update(_RENDER_ZERO)
            return results

        if self.verbose:
            print("  Rendering multi-view images...")

        view_dir = self.render_dir / mesh_id
        view_dir.mkdir(parents=True, exist_ok=True)

        try:
            rendered_paths = render_multiview_images(
                mesh_path, str(view_dir), num_views=8, resolution=512
            )
        except Exception as e:
            if self.verbose:
                print(f"  Rendering failed: {e}")
            results.update(_RENDER_ZERO)
            return results

        if not rendered_paths:
            results.update(_RENDER_ZERO)
            return results

        if self.verbose:
            print("  Computing CLIP scores...")
        results.update(self.clip_evaluator.compute_multiview_clip_score(rendered_paths, prompt))

        if self.verbose:
            print("  Computing multi-view consistency...")
        results['multiview_consistency'] = self.clip_evaluator.compute_multiview_consistency(rendered_paths)

        if self.verbose:
            print("  Computing ImageReward scores...")
        results.update(self.image_reward_evaluator.compute_score(rendered_paths, prompt))

        return results

    def evaluate_batch(self,
                       mesh_list: List[Dict[str, str]],
                       output_name: str = "evaluation_results",
                       skip_rendering: bool = False) -> pd.DataFrame:
        all_results = []

        for item in tqdm(mesh_list, desc="Evaluating meshes", disable=not self.verbose):
            result = self.evaluate_single_mesh(
                mesh_path=item['mesh_path'],
                prompt=item['prompt'],
                mesh_id=item['mesh_id'],
                reference_path=item.get('reference_path'),
                skip_rendering=skip_rendering,
            )
            all_results.append(result)

        df = pd.DataFrame(all_results)

        csv_path  = self.output_dir / f"{output_name}.csv"
        json_path = self.output_dir / f"{output_name}.json"

        df.to_csv(csv_path, index=False)
        if self.verbose:
            print(f"\nSaved CSV results to: {csv_path}")

        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        if self.verbose:
            print(f"Saved JSON results to: {json_path}")

        return df

    def generate_summary_report(self, df: pd.DataFrame, output_name: str = "summary"):
        summary = {}
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

        for col in numeric_cols:
            summary[col] = {
                'mean':   float(df[col].mean()),
                'std':    float(df[col].std()),
                'min':    float(df[col].min()),
                'max':    float(df[col].max()),
                'median': float(df[col].median()),
            }

        summary_path = self.output_dir / f"{output_name}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        if self.verbose:
            print(f"\nSummary statistics saved to: {summary_path}")
            print("\n=== Summary ===")
            print(f"Total meshes evaluated:  {len(df)}")
            print(f"Successfully loaded:     {int(df['load_success'].sum())}")
            print(f"Watertight meshes:       {int(df['is_watertight'].sum())}")
            if 'clip_score_mean' in df.columns:
                print(f"Avg CLIP score:          {df['clip_score_mean'].mean():.3f}")
            if 'multiview_consistency' in df.columns:
                print(f"Avg multiview consist.:  {df['multiview_consistency'].mean():.3f}")
            if 'image_reward_mean' in df.columns:
                print(f"Avg ImageReward:         {df['image_reward_mean'].mean():.3f}")
            if 'face_aspect_ratio_mean' in df.columns:
                print(f"Avg face aspect ratio:   {df['face_aspect_ratio_mean'].mean():.3f}")


if __name__ == "__main__":
    pass