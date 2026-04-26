#!/usr/bin/env python3
"""
ImageReward-based semantic alignment metric.

ImageReward is a reward model trained on human preference data for text-image
alignment. It correlates better with human judgement than raw CLIP-S.

Reference: Xu et al., "ImageReward: Learning and Evaluating Human Preferences
           for Text-to-Image Generation", NeurIPS 2023.

Install: pip install image-reward
"""

import warnings
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


class ImageRewardEvaluator:
    """Score rendered views against a text prompt using ImageReward."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Args:
            cache_dir: Directory to cache the downloaded model weights.
                       Defaults to ~/.cache/ImageReward.
        """
        self.model = None
        self.available = False

        try:
            import ImageReward as RM
        except ImportError:
            warnings.warn(
                "ImageReward not installed. Run: pip install image-reward\n"
                "Scores will be 0 until installed."
            )
            return

        try:
            load_kwargs = {}
            if cache_dir is not None:
                load_kwargs["download_root"] = cache_dir
            self.model = RM.load("ImageReward-v1.0", **load_kwargs)
            self.available = True
        except Exception as e:
            warnings.warn(
                f"ImageReward model failed to load: {e}\n"
                "The model may need to download on first use (~2 GB). "
                "Ensure you have internet access. Scores will be 0."
            )

    def compute_score(self, image_paths: List[str], prompt: str) -> Dict[str, float]:
        """
        Score each rendered view against the text prompt.

        Args:
            image_paths: Paths to rendered view images.
            prompt:      The text prompt used to generate the mesh.

        Returns:
            Dict with image_reward_mean, _std, _min, _max.
            Higher scores indicate better text-to-3D alignment.
        """
        empty = {
            'image_reward_mean': 0.0,
            'image_reward_std':  0.0,
            'image_reward_min':  0.0,
            'image_reward_max':  0.0,
        }

        if not self.available or not image_paths:
            return empty

        scores = []
        for img_path in image_paths:
            if not Path(img_path).exists():
                warnings.warn(f"ImageReward: image not found: {img_path}")
                continue
            try:
                from PIL import Image
                pil_image = Image.open(img_path).convert("RGB")
                result = self.model.score(prompt, pil_image)
                # score() returns a float; guard against unexpected tuple/list
                if isinstance(result, (list, tuple)):
                    result = result[0]
                scores.append(float(result))
            except Exception as e:
                warnings.warn(f"ImageReward failed on {img_path}: {e}")

        if not scores:
            return empty

        return {
            'image_reward_mean': float(np.mean(scores)),
            'image_reward_std':  float(np.std(scores)),
            'image_reward_min':  float(np.min(scores)),
            'image_reward_max':  float(np.max(scores)),
        }