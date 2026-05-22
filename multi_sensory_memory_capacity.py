#!/usr/bin/env python3
"""HyperCOCO multi-sensory memory capacity evaluation.

This script evaluates memory capacity (MC) of learned HyperCBTs using
different sensory streams:

- `mnist`
- `colored_mnist`
- `shapes`
- `mnist_shapes` (combined training/testing stream)
- `gutenberg`
- `wikipedia`
- `dialogue`
- `audio`

Expected input from the first script (`cbt_generation.py`) is a saved fold
results file containing one HyperCBT per fold (key usually `HyperCBT`).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


########################################
# Generic Utility Functions
########################################


def shift_array(arr: np.ndarray, n: int) -> np.ndarray:
    shifted_arr = np.roll(arr, n)
    shifted_arr[:n] = 0
    return shifted_arr


def gen_lag_data(time_series: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(time_series).reshape(-1)
    y = np.zeros((len(x), max_lag), dtype=np.float32)
    for lag in range(1, max_lag + 1):
        y[:, lag - 1] = shift_array(x, lag)
    return x.astype(np.float32), y


def gen_lag_data_embeddings(
    embeddings: np.ndarray,
    num_points: int,
    max_lag: int,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate lagged pairs from embedding values.

    Notes:
    - Works with 1D or multi-dimensional arrays.
    - Multi-dimensional embeddings are flattened into one long sequence,
      then `num_points` values are sampled.
    """
    if seed is not None:
        random.seed(seed)
    flat = np.asarray(embeddings).reshape(-1)
    if num_points > len(flat):
        raise ValueError(
            f"num_points={num_points} is larger than embedding sequence length={len(flat)}."
        )
    indices = random.sample(range(len(flat)), num_points)
    x = flat[indices].astype(np.float32)
    y = np.zeros((num_points, max_lag), dtype=np.float32)
    for lag in range(1, max_lag + 1):
        y[:, lag - 1] = shift_array(x, lag)
    return x, y


def generate_lagged_targets(data: np.ndarray, max_lag: int) -> np.ndarray:
    num_samples, input_dim = data.shape
    lagged = np.zeros((num_samples, max_lag, input_dim), dtype=np.float32)
    for lag in range(1, max_lag + 1):
        lagged[lag:, lag - 1, :] = data[:-lag]
    return lagged.reshape(num_samples, -1)


def safe_normalize_connectivity(matrix: np.ndarray) -> np.ndarray:
    conn = np.asarray(matrix, dtype=np.float32)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(conn))))
    if spectral_radius > 0:
        conn = conn / spectral_radius
    return conn.astype(np.float32)


def split_train_test(
    x: np.ndarray, y: np.ndarray, train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = int(train_ratio * len(x))
    return x[:split], y[:split], x[split:], y[split:]


def safe_pearson_r2(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import pearsonr

    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    r, _ = pearsonr(x, y)
    if np.isnan(r):
        return 0.0
    return float(r**2)


########################################
# Loading HyperCBTs
########################################


def load_fold_results(results_path: str) -> List[Any]:
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find fold results at {path}. Run cbt_generation.py first."
        )
    loaded = np.load(path, allow_pickle=True)

    if isinstance(loaded, np.ndarray):
        if loaded.shape == ():
            obj = loaded.item()
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and "fold_results" in obj:
                return list(obj["fold_results"])
            return [obj]
        return list(loaded.tolist())

    if isinstance(loaded, list):
        return loaded
    return [loaded]


def extract_cbts(fold_results: Sequence[Any], cbt_key: str = "auto") -> List[np.ndarray]:
    candidate_keys = ["HyperCBT", "hypercbt", "CBT", "cbt"]
    cbts: List[np.ndarray] = []

    for idx, fold in enumerate(fold_results):
        cbt = None
        if isinstance(fold, dict):
            if cbt_key != "auto":
                cbt = fold.get(cbt_key)
            else:
                for key in candidate_keys:
                    if key in fold:
                        cbt = fold[key]
                        break
        elif isinstance(fold, np.ndarray):
            cbt = fold

        if cbt is None:
            raise KeyError(
                f"Could not find CBT in fold index {idx}. "
                f"Available key(s): {list(fold.keys()) if isinstance(fold, dict) else type(fold)}"
            )
        cbts.append(np.asarray(cbt, dtype=np.float32))

    if not cbts:
        raise ValueError("No CBTs were extracted from the provided fold results.")
    return cbts


########################################
# Image Utilities
########################################


def crop_and_resize(images, target_size: int = 15):
    """Crop foreground and resize grayscale images (MNIST-like)."""
    import torch
    from torch.nn.functional import interpolate

    cropped_images = []
    for img in images:
        img_np = img.numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
        non_zero_rows = np.where(img_np.sum(axis=1) > 0)[0]
        non_zero_cols = np.where(img_np.sum(axis=0) > 0)[0]

        if len(non_zero_rows) > 0 and len(non_zero_cols) > 0:
            top, bottom = non_zero_rows[0], non_zero_rows[-1] + 1
            left, right = non_zero_cols[0], non_zero_cols[-1] + 1
            cropped = img_np[top:bottom, left:right]
            cropped_tensor = torch.tensor(cropped).unsqueeze(0).unsqueeze(0)
            resized = interpolate(
                cropped_tensor,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
            cropped_images.append(resized.squeeze(0).squeeze(0))
        else:
            cropped_images.append(torch.zeros(target_size, target_size))
    return torch.stack(cropped_images)


def crop_and_resize_color(images, target_size: int = 15):
    """Crop foreground and resize RGB images."""
    import torch
    from torch.nn.functional import interpolate

    cropped_images = []
    for img in images:
        img_np = img.numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
        non_zero_rows = np.where(img_np.sum(axis=(1, 2)) > 0)[0]
        non_zero_cols = np.where(img_np.sum(axis=(0, 2)) > 0)[0]

        if len(non_zero_rows) > 0 and len(non_zero_cols) > 0:
            top, bottom = non_zero_rows[0], non_zero_rows[-1] + 1
            left, right = non_zero_cols[0], non_zero_cols[-1] + 1
            cropped = img_np[top:bottom, left:right, :]
            cropped_tensor = torch.tensor(cropped).permute(2, 0, 1).unsqueeze(0)
            resized = interpolate(
                cropped_tensor,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
            cropped_images.append(resized.squeeze(0))
        else:
            cropped_images.append(torch.zeros(3, target_size, target_size))
    return torch.stack(cropped_images)


def load_shapes_data(
    dataset_path: str,
    shape_classes: Sequence[str],
    samples_per_class: int,
    target_size: int,
    grayscale: bool,
    seed: int,
) -> Tuple[np.ndarray, List[str]]:
    """Load shape images, crop foreground, resize, and return tensor-like arrays."""
    from PIL import Image
    import torch
    from torchvision import transforms

    rng = random.Random(seed)
    transform = transforms.Compose([transforms.ToTensor()])

    shapes_data = []
    labels = []

    for shape in shape_classes:
        shape_folder = Path(dataset_path) / shape
        if not shape_folder.exists():
            raise FileNotFoundError(f"Shape class folder not found: {shape_folder}")

        image_files = [
            f
            for f in shape_folder.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        if not image_files:
            raise ValueError(f"No images found in: {shape_folder}")

        chosen = rng.sample(image_files, min(samples_per_class, len(image_files)))
        batch = []

        for file_path in chosen:
            img = Image.open(file_path).convert("L" if grayscale else "RGB")
            img_np = np.asarray(img)
            mask = (img_np != 0) if grayscale else np.any(img_np != 0, axis=-1)
            non_zero_rows = np.where(mask.sum(axis=1) > 0)[0]
            non_zero_cols = np.where(mask.sum(axis=0) > 0)[0]

            if len(non_zero_rows) > 0 and len(non_zero_cols) > 0:
                top, bottom = non_zero_rows[0], non_zero_rows[-1] + 1
                left, right = non_zero_cols[0], non_zero_cols[-1] + 1
                cropped = (
                    img_np[top:bottom, left:right]
                    if grayscale
                    else img_np[top:bottom, left:right, :]
                )
                cropped_pil = (
                    Image.fromarray(cropped)
                    if grayscale
                    else Image.fromarray(cropped, mode="RGB")
                )
                resized = cropped_pil.resize((target_size, target_size))
                batch.append(transform(resized))
            else:
                batch.append(
                    torch.zeros(1, target_size, target_size)
                    if grayscale
                    else torch.zeros(3, target_size, target_size)
                )

        shapes_data.append(torch.stack(batch))
        labels.extend([shape] * len(batch))

    return torch.cat(shapes_data).numpy(), labels


########################################
# Memory Capacity Core
########################################


def build_esn(connectivity_matrix: np.ndarray):
    """Create ESN with HyperCOCO settings used in notebook experiments."""
    try:
        from echoes import ESNRegressor
    except ImportError as exc:
        raise ImportError(
            "The `echoes` package is required to run memory-capacity evaluation. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    return ESNRegressor(
        spectral_radius=0.9,
        input_scaling=1,
        leak_rate=0.9,
        bias=0,
        W=connectivity_matrix,
        random_state=42,
    )


def compute_memory_capacity_corrcoef(
    connectivity_matrix: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_lag: int,
    input_dim: int,
) -> Tuple[List[float], float]:
    """MC using lag-wise corrcoef over predicted vs target features."""
    esn = build_esn(connectivity_matrix)
    esn.fit(x_train, y_train)
    y_pred = esn.predict(x_test)

    y_pred = np.asarray(y_pred).reshape(-1, max_lag, input_dim)
    y_true = np.asarray(y_test).reshape(-1, max_lag, input_dim)

    per_lag_r2 = []
    total_memory_capacity = 0.0

    for lag in range(max_lag):
        pred = y_pred[:, lag, :]
        target = y_true[:, lag, :]
        corr_matrix = np.corrcoef(pred.T, target.T)
        corr_matrix_upper = np.triu(corr_matrix, 1)
        corr_matrix_upper = corr_matrix_upper[~np.isnan(corr_matrix_upper)]
        if corr_matrix_upper.size == 0:
            r2 = 0.0
        else:
            corr = float(np.mean(np.abs(corr_matrix_upper)))
            r2 = corr**2
        per_lag_r2.append(r2)
        total_memory_capacity += r2

    return per_lag_r2, float(total_memory_capacity)


def compute_memory_capacity_pearson(
    connectivity_matrix: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_lag: int,
) -> Tuple[List[float], float]:
    """MC using lag-wise Pearson r^2."""
    esn = build_esn(connectivity_matrix)
    esn.fit(x_train, y_train)
    y_pred = np.asarray(esn.predict(x_test))
    y_true = np.asarray(y_test)

    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]

    per_lag_r2 = []
    total_memory_capacity = 0.0
    for i in range(max_lag):
        r2 = safe_pearson_r2(y_true[:, i], y_pred[:, i])
        per_lag_r2.append(r2)
        total_memory_capacity += r2
    return per_lag_r2, float(total_memory_capacity)


def evaluate_cbts(
    cbts: Sequence[np.ndarray],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_lag: int,
    metric: str,
    input_dim: int | None = None,
    label: str = "",
) -> List[Dict[str, Any]]:
    results = []
    for idx, cbt in enumerate(cbts, start=1):
        connectivity_matrix = safe_normalize_connectivity(cbt)
        if metric == "corrcoef":
            if input_dim is None:
                raise ValueError("input_dim is required for corrcoef metric.")
            per_lag, total_mc = compute_memory_capacity_corrcoef(
                connectivity_matrix,
                x_train.astype(np.float32),
                y_train.astype(np.float32),
                x_test.astype(np.float32),
                y_test.astype(np.float32),
                max_lag=max_lag,
                input_dim=input_dim,
            )
        elif metric == "pearson":
            per_lag, total_mc = compute_memory_capacity_pearson(
                connectivity_matrix,
                x_train.astype(np.float32),
                y_train.astype(np.float32),
                x_test.astype(np.float32),
                y_test.astype(np.float32),
                max_lag=max_lag,
            )
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        msg_label = f"{label} " if label else ""
        print(f"{msg_label}CBT #{idx:02d} Memory Capacity: {total_mc:.6f}")
        results.append(
            {
                "cbt_index": idx,
                "memory_capacity": total_mc,
                "per_lag_r2": per_lag,
            }
        )
    return results


########################################
# Modality-Specific Runners
########################################


def run_mnist(cbts: Sequence[np.ndarray], args: argparse.Namespace) -> List[Dict[str, Any]]:
    import torch
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(
        root=args.mnist_data_root,
        train=True,
        download=True,
        transform=transform,
    )

    num_samples = args.mnist_num_samples
    max_lag = args.mnist_max_lag
    images = mnist_train.data[:num_samples].float() / 255.0
    images = crop_and_resize(images, target_size=args.image_size)
    mnist_x = images.view(num_samples, -1).numpy().astype(np.float32)
    mnist_y = generate_lagged_targets(mnist_x, max_lag)
    x_train, y_train, x_test, y_test = split_train_test(mnist_x, mnist_y, args.train_ratio)

    return evaluate_cbts(
        cbts,
        x_train,
        y_train,
        x_test,
        y_test,
        max_lag=max_lag,
        metric="corrcoef",
        input_dim=mnist_x.shape[1],
        label="MNIST",
    )


def run_colored_mnist(
    cbts: Sequence[np.ndarray], args: argparse.Namespace
) -> List[Dict[str, Any]]:
    import torch

    if not args.colored_mnist_path:
        raise ValueError("`--colored-mnist-path` is required for modality `colored_mnist`.")
    path = Path(args.colored_mnist_path)
    if not path.exists():
        raise FileNotFoundError(f"Colored MNIST file not found: {path}")

    loaded_images = np.load(path, allow_pickle=True)
    num_samples = min(args.colored_num_samples, len(loaded_images))
    max_lag = args.colored_max_lag

    images = crop_and_resize_color(
        torch.tensor(loaded_images[:num_samples]).float() / 255.0,
        target_size=args.image_size,
    )
    color_x = images.view(images.shape[0], -1).numpy().astype(np.float32)
    color_y = generate_lagged_targets(color_x, max_lag)
    x_train, y_train, x_test, y_test = split_train_test(color_x, color_y, args.train_ratio)

    return evaluate_cbts(
        cbts,
        x_train,
        y_train,
        x_test,
        y_test,
        max_lag=max_lag,
        metric="corrcoef",
        input_dim=color_x.shape[1],
        label="Colored-MNIST",
    )


def run_shapes(cbts: Sequence[np.ndarray], args: argparse.Namespace) -> List[Dict[str, Any]]:
    if not args.shapes_dataset_path:
        raise ValueError("`--shapes-dataset-path` is required for modality `shapes`.")
    shape_classes = [x.strip() for x in args.shapes_classes.split(",") if x.strip()]
    if not shape_classes:
        raise ValueError("`--shapes-classes` cannot be empty.")

    shapes_x, _ = load_shapes_data(
        dataset_path=args.shapes_dataset_path,
        shape_classes=shape_classes,
        samples_per_class=args.shapes_samples_per_class,
        target_size=args.image_size,
        grayscale=True,
        seed=args.shapes_seed,
    )
    num_samples = len(shapes_x)
    max_lag = args.shapes_max_lag

    shapes_x = shapes_x.reshape(num_samples, -1).astype(np.float32)
    shapes_y = generate_lagged_targets(shapes_x, max_lag)
    x_train, y_train, x_test, y_test = split_train_test(
        shapes_x, shapes_y, args.train_ratio
    )

    return evaluate_cbts(
        cbts,
        x_train,
        y_train,
        x_test,
        y_test,
        max_lag=max_lag,
        metric="corrcoef",
        input_dim=shapes_x.shape[1],
        label="Shapes",
    )


def run_mnist_shapes(cbts: Sequence[np.ndarray], args: argparse.Namespace) -> List[Dict[str, Any]]:
    import torch
    from torchvision import datasets, transforms

    if not args.shapes_dataset_path:
        raise ValueError("`--shapes-dataset-path` is required for modality `mnist_shapes`.")

    # MNIST branch
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(
        root=args.mnist_data_root,
        train=True,
        download=True,
        transform=transform,
    )
    mnist_n = args.combined_mnist_num_samples
    max_lag = args.combined_max_lag
    mnist_imgs = mnist_train.data[:mnist_n].float() / 255.0
    mnist_imgs = crop_and_resize(mnist_imgs, target_size=args.image_size)
    mnist_x = mnist_imgs.view(mnist_n, -1).numpy().astype(np.float32)
    mnist_y = generate_lagged_targets(mnist_x, max_lag)
    x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist = split_train_test(
        mnist_x, mnist_y, args.train_ratio
    )

    # Shapes branch
    shape_classes = [x.strip() for x in args.shapes_classes.split(",") if x.strip()]
    shapes_x, _ = load_shapes_data(
        dataset_path=args.shapes_dataset_path,
        shape_classes=shape_classes,
        samples_per_class=args.combined_shapes_samples_per_class,
        target_size=args.image_size,
        grayscale=True,
        seed=args.shapes_seed,
    )
    shapes_x = shapes_x.reshape(len(shapes_x), -1).astype(np.float32)
    shapes_y = generate_lagged_targets(shapes_x, max_lag)
    x_train_shapes, y_train_shapes, x_test_shapes, y_test_shapes = split_train_test(
        shapes_x, shapes_y, args.train_ratio
    )

    if x_train_mnist.shape[1] != x_train_shapes.shape[1]:
        raise ValueError(
            "MNIST and Shapes feature dimensions must match for combined modality."
        )

    # Same design as notebook: stack both streams in one training/testing set.
    x_train_combined = np.vstack([x_train_mnist, x_train_shapes]).astype(np.float32)
    y_train_combined = np.vstack([y_train_mnist, y_train_shapes]).astype(np.float32)
    x_test_combined = np.vstack([x_test_mnist, x_test_shapes]).astype(np.float32)
    y_test_combined = np.vstack([y_test_mnist, y_test_shapes]).astype(np.float32)

    return evaluate_cbts(
        cbts,
        x_train_combined,
        y_train_combined,
        x_test_combined,
        y_test_combined,
        max_lag=max_lag,
        metric="corrcoef",
        # Important: input_dim is per sample (not MNIST+Shapes sum).
        input_dim=x_train_mnist.shape[1],
        label="MNIST+Shapes",
    )


def load_embedding_array(path: str) -> np.ndarray:
    """Load embeddings from .npy/.npz or object file.

    For keyed embeddings (e.g., gensim-like object with `key_to_index`),
    a flattened sampled sequence is constructed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embedding file not found: {p}")

    loaded = np.load(p, allow_pickle=True)
    obj = loaded.item() if isinstance(loaded, np.ndarray) and loaded.shape == () else loaded

    if hasattr(obj, "key_to_index"):
        # Wiki-like keyed vectors
        words = list(obj.key_to_index.keys())
        if not words:
            raise ValueError("Embedding object has empty vocabulary.")
        rng = random.Random(43)
        samples = np.array([obj[rng.choice(words)] for _ in range(100)])
        return samples.flatten()

    return np.asarray(obj)


def run_embedding_modality(
    cbts: Sequence[np.ndarray],
    embedding_path: str,
    max_lag: int,
    train_points: int,
    test_points: int,
    label: str,
) -> List[Dict[str, Any]]:
    embeddings = load_embedding_array(embedding_path)
    x_train_np, y_train_np = gen_lag_data_embeddings(
        embeddings, train_points, max_lag, seed=41
    )
    x_test_np, y_test_np = gen_lag_data_embeddings(
        embeddings, test_points, max_lag, seed=42
    )

    # 1D sequence -> ESN input with one feature.
    x_train = x_train_np.reshape(-1, 1).astype(np.float32)
    x_test = x_test_np.reshape(-1, 1).astype(np.float32)
    y_train = y_train_np.astype(np.float32)
    y_test = y_test_np.astype(np.float32)

    return evaluate_cbts(
        cbts,
        x_train,
        y_train,
        x_test,
        y_test,
        max_lag=max_lag,
        metric="pearson",
        label=label,
    )


def run_audio(
    cbts: Sequence[np.ndarray],
    audio_path: str,
    max_lag: int,
    mfcc_index: int,
) -> List[Dict[str, Any]]:
    import librosa

    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    audio_signal, sr = librosa.load(str(p), sr=None)
    mfcc = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=13).T
    x, y = gen_lag_data(mfcc[:, mfcc_index], max_lag)
    x = x.reshape(-1, 1).astype(np.float32)
    y = y.astype(np.float32)
    x_train, y_train, x_test, y_test = split_train_test(x, y, 0.8)

    return evaluate_cbts(
        cbts,
        x_train,
        y_train,
        x_test,
        y_test,
        max_lag=max_lag,
        metric="pearson",
        label=f"Audio({p.name})",
    )


########################################
# CLI
########################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HyperCOCO multi-sensory memory capacity evaluation."
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="outputs/hypercbt_fold_results.npy",
        help="Path to saved fold results from cbt_generation.py.",
    )
    parser.add_argument(
        "--cbt-key",
        type=str,
        default="auto",
        help="CBT key in each fold dict (default: auto-detect).",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="mnist",
        help=(
            "Comma-separated modalities: "
            "mnist,colored_mnist,shapes,mnist_shapes,gutenberg,wikipedia,dialogue,audio"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to save all MC outputs as JSON.",
    )

    # Shared image settings
    parser.add_argument("--image-size", type=int, default=15, help="Target image size.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")

    # MNIST
    parser.add_argument("--mnist-data-root", type=str, default="./data")
    parser.add_argument("--mnist-num-samples", type=int, default=100)
    parser.add_argument("--mnist-max-lag", type=int, default=20)

    # Colored MNIST (or any pre-saved RGB image tensor)
    parser.add_argument("--colored-mnist-path", type=str, default="")
    parser.add_argument("--colored-num-samples", type=int, default=1000)
    parser.add_argument("--colored-max-lag", type=int, default=20)

    # Shapes
    parser.add_argument("--shapes-dataset-path", type=str, default="")
    parser.add_argument("--shapes-classes", type=str, default="Circle,Square,Triangle")
    parser.add_argument("--shapes-samples-per-class", type=int, default=30)
    parser.add_argument("--shapes-max-lag", type=int, default=20)
    parser.add_argument("--shapes-seed", type=int, default=132)

    # Combined MNIST+Shapes
    parser.add_argument("--combined-mnist-num-samples", type=int, default=90)
    parser.add_argument("--combined-shapes-samples-per-class", type=int, default=30)
    parser.add_argument("--combined-max-lag", type=int, default=10)

    # Embeddings
    parser.add_argument("--gutenberg-embeddings-path", type=str, default="")
    parser.add_argument("--wikipedia-embeddings-path", type=str, default="")
    parser.add_argument("--dialogue-embeddings-path", type=str, default="")
    parser.add_argument("--embedding-max-lag", type=int, default=20)
    parser.add_argument("--embedding-train-points", type=int, default=1000)
    parser.add_argument("--embedding-test-points", type=int, default=200)

    # Audio
    parser.add_argument(
        "--audio-paths",
        type=str,
        default="",
        help="Comma-separated audio file paths.",
    )
    parser.add_argument("--audio-max-lag", type=int, default=20)
    parser.add_argument("--audio-mfcc-index", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fold_results = load_fold_results(args.results_path)
    cbts = extract_cbts(fold_results, cbt_key=args.cbt_key)
    print(f"Loaded {len(cbts)} CBT(s) from {args.results_path}")

    modalities = [m.strip().lower() for m in args.modalities.split(",") if m.strip()]
    if not modalities:
        raise ValueError("No modality selected. Use --modalities.")

    all_results: Dict[str, Any] = {
        "results_path": args.results_path,
        "n_cbts": len(cbts),
        "modalities": {},
    }

    for modality in modalities:
        print(f"\n=== Running modality: {modality} ===")

        if modality == "mnist":
            all_results["modalities"]["mnist"] = run_mnist(cbts, args)

        elif modality == "colored_mnist":
            all_results["modalities"]["colored_mnist"] = run_colored_mnist(cbts, args)

        elif modality == "shapes":
            all_results["modalities"]["shapes"] = run_shapes(cbts, args)

        elif modality == "mnist_shapes":
            all_results["modalities"]["mnist_shapes"] = run_mnist_shapes(cbts, args)

        elif modality == "gutenberg":
            if not args.gutenberg_embeddings_path:
                raise ValueError(
                    "`--gutenberg-embeddings-path` is required for `gutenberg`."
                )
            all_results["modalities"]["gutenberg"] = run_embedding_modality(
                cbts=cbts,
                embedding_path=args.gutenberg_embeddings_path,
                max_lag=args.embedding_max_lag,
                train_points=args.embedding_train_points,
                test_points=args.embedding_test_points,
                label="Gutenberg",
            )

        elif modality == "wikipedia":
            if not args.wikipedia_embeddings_path:
                raise ValueError(
                    "`--wikipedia-embeddings-path` is required for `wikipedia`."
                )
            all_results["modalities"]["wikipedia"] = run_embedding_modality(
                cbts=cbts,
                embedding_path=args.wikipedia_embeddings_path,
                max_lag=args.embedding_max_lag,
                train_points=max(args.embedding_train_points, 10000),
                test_points=max(args.embedding_test_points, 2000),
                label="Wikipedia",
            )

        elif modality == "dialogue":
            if not args.dialogue_embeddings_path:
                raise ValueError(
                    "`--dialogue-embeddings-path` is required for `dialogue`."
                )
            all_results["modalities"]["dialogue"] = run_embedding_modality(
                cbts=cbts,
                embedding_path=args.dialogue_embeddings_path,
                max_lag=args.embedding_max_lag,
                train_points=max(args.embedding_train_points, 10000),
                test_points=max(args.embedding_test_points, 2000),
                label="Dialogue",
            )

        elif modality == "audio":
            if not args.audio_paths:
                raise ValueError("`--audio-paths` is required for modality `audio`.")
            audio_files = [p.strip() for p in args.audio_paths.split(",") if p.strip()]
            audio_results: Dict[str, Any] = {}
            for audio_file in audio_files:
                audio_results[Path(audio_file).name] = run_audio(
                    cbts=cbts,
                    audio_path=audio_file,
                    max_lag=args.audio_max_lag,
                    mfcc_index=args.audio_mfcc_index,
                )
            all_results["modalities"]["audio"] = audio_results

        else:
            raise ValueError(
                f"Unknown modality `{modality}`. "
                "Use one of: mnist,colored_mnist,shapes,mnist_shapes,gutenberg,wikipedia,dialogue,audio"
            )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved MC results to: {out_path}")


if __name__ == "__main__":
    main()
