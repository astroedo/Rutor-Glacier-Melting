#!/usr/bin/env python3
"""
Rutor Glacier - Inference / Prediction
Apply trained models to GeoTIFF composites for temporal glacier classification.

Usage:
    python src/models/predict.py --input data/composites/ --output results/
    python src/models/predict.py --input data/composites/ --output results/ --model cnn
    python src/models/predict.py --input data/composites/ --output results/ --model mlp
    python src/models/predict.py --input data/composites/ --output results/ --model rf
"""

import argparse
import glob
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = {0: 'Clean Ice', 1: 'Debris Ice', 2: 'Water', 3: 'Vegetation', 4: 'Rock'}
FEATURE_NAMES = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'Thermal', 'NDSI', 'NDVI', 'NDWI']
N_FEATURES = len(FEATURE_NAMES)
N_CLASSES = len(CLASS_NAMES)
PIXEL_AREA_KM2 = 0.0009  # 30 m x 30 m = 900 m²

TIME_PERIODS = [
    {'name': '1984_1988', 'mid_year': 1986},
    {'name': '1989_1993', 'mid_year': 1991},
    {'name': '1994_1998', 'mid_year': 1996},
    {'name': '1999_2003', 'mid_year': 2001},
    {'name': '2004_2008', 'mid_year': 2006},
    {'name': '2009_2013', 'mid_year': 2011},
    {'name': '2014_2018', 'mid_year': 2016},
    {'name': '2019_2024', 'mid_year': 2021},
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Apply trained model to GeoTIFF composites')
    parser.add_argument('--input', type=str, required=True,
                        help='Directory containing composite GeoTIFFs')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for classification maps and plots')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'mlp', 'rf'],
                        help='Which trained model to use (default: cnn)')
    parser.add_argument('--model-dir', type=str, default='Result',
                        help='Directory containing saved models and scaler (default: Result)')
    parser.add_argument('--confidence', type=float, default=0.6,
                        help='Min confidence; below this the pixel defaults to Rock (default: 0.6)')
    parser.add_argument('--ndsi-threshold', type=float, default=0.2,
                        help='NDSI floor for Clean Ice; below this -> Rock (default: 0.2)')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_model(model_dir, model_type):
    """Load a trained model and its scaler from disk."""
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    if model_type == 'cnn':
        import tensorflow as tf
        model_path = os.path.join(model_dir, 'cnn_model.keras')
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded CNN from {model_path}")
    elif model_type == 'mlp':
        model_path = os.path.join(model_dir, 'mlp_model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Loaded MLP from {model_path}")
    elif model_type == 'rf':
        model_path = os.path.join(model_dir, 'rf_model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Loaded RF from {model_path}")

    return model, scaler


def load_gee_composite(filepath):
    """Load a multi-band GeoTIFF composite exported from GEE."""
    with rasterio.open(filepath) as src:
        data = src.read()                        # (bands, H, W)
        data = np.moveaxis(data, 0, -1)          # -> (H, W, bands)
        return data, src.transform, src.crs, src.profile


def save_classification_geotiff(classification, profile, filepath):
    """Save a classification map as a single-band GeoTIFF."""
    out_profile = profile.copy()
    out_profile.update(dtype='int8', count=1, nodata=-1)
    with rasterio.open(filepath, 'w', **out_profile) as dst:
        dst.write(classification.astype(np.int8), 1)
    print(f"  Saved: {filepath}")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify_composite(model, scaler, data, model_type,
                       confidence_threshold=0.6, ndsi_threshold=0.2):
    """Apply a trained model to an entire raster composite.

    Two-stage post-processing to reduce ice overestimation:
      1. Confidence threshold: low-confidence pixels default to Rock (4).
      2. NDSI physical constraint: Clean Ice (0) with NDSI < threshold -> Rock.
    """
    h, w, bands = data.shape
    if bands != N_FEATURES:
        print(f"  Expected {N_FEATURES} bands, got {bands}. Skipping.")
        return None

    flat = data.reshape(-1, N_FEATURES)

    # Mask out nodata pixels (all zeros or NaN)
    valid = ~(np.isnan(flat).any(axis=1) | (flat == 0).all(axis=1))
    classification = np.full(flat.shape[0], -1, dtype=np.int8)

    if valid.sum() == 0:
        return classification.reshape(h, w)

    # RF uses raw features; CNN and MLP use scaled features
    if model_type == 'rf':
        features = flat[valid]
    else:
        features = scaler.transform(flat[valid])

    # Predict probabilities
    if model_type == 'cnn':
        cnn_input = features.reshape(-1, N_FEATURES, 1)
        proba = model.predict(cnn_input, batch_size=1024, verbose=0)
    else:  # mlp or rf
        proba = model.predict_proba(features)

    preds = np.argmax(proba, axis=1)
    max_proba = np.max(proba, axis=1)

    # Stage 1 — low-confidence pixels -> Rock (4)
    preds[max_proba < confidence_threshold] = 4

    # Stage 2 — NDSI physical constraint on Clean Ice
    ndsi_vals = flat[valid, 7]  # NDSI is band index 7
    clean_ice_low_ndsi = (preds == 0) & (ndsi_vals < ndsi_threshold)
    preds[clean_ice_low_ndsi] = 4

    classification[valid] = preds
    return classification.reshape(h, w)


# ---------------------------------------------------------------------------
# Period matching
# ---------------------------------------------------------------------------
def extract_period_info(filename):
    """Match a composite filename to a known time period."""
    base = os.path.splitext(os.path.basename(filename))[0]
    for period in TIME_PERIODS:
        if period['name'] in base:
            return period
    return None


# ---------------------------------------------------------------------------
# Temporal evolution plot
# ---------------------------------------------------------------------------
def plot_temporal(temporal, model_name, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: total ice
    ax = axes[0]
    ax.plot(temporal['years'], temporal['total_ice_km2'],
            'o-', lw=2.5, ms=8, color='#d62728', label=f'{model_name} Total Ice')
    ax.set_xlabel('Year', fontsize=13)
    ax.set_ylabel('Total Ice Area (km\u00b2)', fontsize=13)
    ax.set_title(f'Ice Area Evolution ({model_name})\nRutor Glacier (1984\u20132024)',
                 fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Right: clean vs debris breakdown
    ax = axes[1]
    ax.plot(temporal['years'], temporal['clean_ice_km2'],
            'o-', lw=2, ms=7, color='deepskyblue', label='Clean Ice')
    ax.plot(temporal['years'], temporal['debris_ice_km2'],
            's-', lw=2, ms=7, color='sienna', label='Debris Ice')
    ax.plot(temporal['years'], temporal['total_ice_km2'],
            '^-', lw=2, ms=7, color='#d62728', label='Total Ice')
    ax.set_xlabel('Year', fontsize=13)
    ax.set_ylabel('Area (km\u00b2)', fontsize=13)
    ax.set_title(f'{model_name} Ice Cover Breakdown\nRutor Glacier (1984\u20132024)',
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_temporal_evolution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Summary table
    print(f"\n=== {model_name} ICE AREA EVOLUTION ===")
    print(f"{'Year':>6} {'Clean (km2)':>12} {'Debris (km2)':>13} {'Total (km2)':>12}")
    print('-' * 48)
    for i, yr in enumerate(temporal['years']):
        print(f"{yr:6d} {temporal['clean_ice_km2'][i]:12.2f} "
              f"{temporal['debris_ice_km2'][i]:13.2f} "
              f"{temporal['total_ice_km2'][i]:12.2f}")

    if temporal['total_ice_km2'][0] > 0:
        loss_pct = ((1 - temporal['total_ice_km2'][-1] /
                     temporal['total_ice_km2'][0]) * 100)
        print(f"\nTotal ice loss: {loss_pct:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Resolve paths relative to project root
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_dir = os.path.join(project_root, args.input)
    output_dir = os.path.join(project_root, args.output)
    model_dir = os.path.join(project_root, args.model_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load model + scaler
    print(f"Loading {args.model.upper()} model from {model_dir}/ ...")
    model, scaler = load_model(model_dir, args.model)

    # Discover composite GeoTIFFs
    tif_files = sorted(glob.glob(os.path.join(input_dir, '*.tif')))
    if not tif_files:
        print(f"No .tif files found in {input_dir}")
        return
    print(f"Found {len(tif_files)} composite(s)\n")

    # Classify each composite
    temporal = {
        'years': [], 'total_ice_km2': [],
        'clean_ice_km2': [], 'debris_ice_km2': [],
    }

    for tif_path in tif_files:
        period = extract_period_info(tif_path)
        basename = os.path.basename(tif_path)
        print(f"Processing {basename} ...")

        data, transform, crs, profile = load_gee_composite(tif_path)
        print(f"  Shape: {data.shape}")

        cmap = classify_composite(
            model, scaler, data, args.model,
            args.confidence, args.ndsi_threshold,
        )
        if cmap is None:
            continue

        # Ice area statistics
        clean_px = np.sum(cmap == 0)
        debris_px = np.sum(cmap == 1)
        clean_km2 = clean_px * PIXEL_AREA_KM2
        debris_km2 = debris_px * PIXEL_AREA_KM2
        total_km2 = clean_km2 + debris_km2

        print(f"  Total ice: {total_km2:.2f} km2  "
              f"(clean: {clean_km2:.2f}, debris: {debris_km2:.2f})")

        if period:
            temporal['years'].append(period['mid_year'])
            temporal['total_ice_km2'].append(total_km2)
            temporal['clean_ice_km2'].append(clean_km2)
            temporal['debris_ice_km2'].append(debris_km2)

        # Save classification GeoTIFF
        out_name = (f"{args.model.upper()}_Classification_"
                    f"{os.path.splitext(basename)[0]}.tif")
        save_classification_geotiff(
            cmap, profile, os.path.join(output_dir, out_name))

    # Temporal evolution plot
    if len(temporal['years']) >= 2:
        plot_temporal(temporal, args.model.upper(), output_dir)

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == '__main__':
    main()
