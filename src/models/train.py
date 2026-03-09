#!/usr/bin/env python3
"""
Rutor Glacier - Model Training
Train CNN, MLP, and/or RF models for 5-class glacier land cover classification.

Usage:
    python src/models/train.py --model cnn --epochs 200
    python src/models/train.py --model mlp
    python src/models/train.py --model rf
    python src/models/train.py --model all --epochs 200
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, f1_score
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = {0: 'Clean Ice', 1: 'Debris Ice', 2: 'Water', 3: 'Vegetation', 4: 'Rock'}
FEATURE_NAMES = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'Thermal', 'NDSI', 'NDVI', 'NDWI']
N_CLASSES = len(CLASS_NAMES)
N_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train glacier classification models (CNN / MLP / RF)')
    parser.add_argument('--model', type=str, default='all',
                        choices=['cnn', 'mlp', 'rf', 'all'],
                        help='Model to train (default: all)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Max training epochs for CNN (default: 200)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for CNN (default: 32)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing CSV datasets (default: data)')
    parser.add_argument('--output-dir', type=str, default='Result',
                        help='Directory for saved models and plots (default: Result)')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(data_dir):
    train_file = os.path.join(data_dir, 'Training_Set_75_Percent.csv')
    test_file = os.path.join(data_dir, 'Testing_Set_25_Percent.csv')

    print("Loading training and testing data...")
    train_data = pd.read_csv(train_file).dropna()
    test_data = pd.read_csv(test_file).dropna()

    X_train = train_data[FEATURE_NAMES].values
    y_train = train_data['class'].values
    X_test = test_data[FEATURE_NAMES].values
    y_test = test_data['class'].values

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print(f"Features: {N_FEATURES} | Classes: {N_CLASSES}")
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------
def build_cnn(input_shape, n_classes):
    """1D-CNN for spectral classification."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        # Block 1
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu',
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.25),
        # Block 2
        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.25),
        # Classifier head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train_cnn(X_train_scaled, y_train, X_test_scaled, y_test,
              epochs, batch_size, output_dir):
    from tensorflow import keras
    from tensorflow.keras import callbacks as cb

    # Reshape for 1D-CNN: (samples, bands, 1 channel)
    X_train_cnn = X_train_scaled.reshape(-1, N_FEATURES, 1)
    X_test_cnn = X_test_scaled.reshape(-1, N_FEATURES, 1)
    y_train_oh = keras.utils.to_categorical(y_train, N_CLASSES)
    y_test_oh = keras.utils.to_categorical(y_test, N_CLASSES)

    model = build_cnn(input_shape=(N_FEATURES, 1), n_classes=N_CLASSES)
    model.summary()

    early_stop = cb.EarlyStopping(monitor='val_loss', patience=30,
                                  restore_best_weights=True)
    reduce_lr = cb.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                     patience=10, min_lr=1e-6)

    history = model.fit(
        X_train_cnn, y_train_oh,
        validation_data=(X_test_cnn, y_test_oh),
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    # Save training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('CNN Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('CNN Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cnn_training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Save model
    model.save(os.path.join(output_dir, 'cnn_model.keras'))
    print(f"CNN model saved to {output_dir}/cnn_model.keras")

    # Evaluate
    y_pred_proba = model.predict(X_test_cnn)
    y_pred = np.argmax(y_pred_proba, axis=1)
    train_pred = np.argmax(model.predict(X_train_cnn), axis=1)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, y_pred)

    return {
        'name': 'CNN', 'train_acc': train_acc, 'test_acc': test_acc,
        'preds': y_pred, 'proba': y_pred_proba,
    }


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------
def train_mlp(X_train_scaled, y_train, X_test_scaled, y_test, output_dir):
    architectures = {
        'MLP Single (128)': (128,),
        'MLP Double (128-64)': (128, 64),
        'MLP Triple (128-64-32)': (128, 64, 32),
    }

    best_acc = 0
    best_model = None
    best_name = None
    best_preds = None

    for name, layers in architectures.items():
        print(f"  Training {name}...", end=' ')
        model = MLPClassifier(
            hidden_layer_sizes=layers, activation='relu', solver='adam',
            alpha=0.001, learning_rate_init=0.001, max_iter=1000,
            random_state=42, early_stopping=True,
        )
        model.fit(X_train_scaled, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
        y_pred = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"Train: {train_acc:.3f} | Test: {test_acc:.3f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model
            best_name = name
            best_preds = y_pred

    # Save best MLP
    with open(os.path.join(output_dir, 'mlp_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    print(f"  Best MLP: {best_name} -> saved to {output_dir}/mlp_model.pkl")

    proba = best_model.predict_proba(X_test_scaled)
    train_acc = accuracy_score(y_train, best_model.predict(X_train_scaled))

    return {
        'name': best_name, 'train_acc': train_acc, 'test_acc': best_acc,
        'preds': best_preds, 'proba': proba,
    }


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------
def train_rf(X_train, y_train, X_test, y_test, output_dir):
    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)
    proba = model.predict_proba(X_test)

    with open(os.path.join(output_dir, 'rf_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    print(f"  RF Train: {train_acc:.3f} | Test: {test_acc:.3f}")
    print(f"  Saved to {output_dir}/rf_model.pkl")

    return {
        'name': 'Random Forest', 'train_acc': train_acc, 'test_acc': test_acc,
        'preds': y_pred, 'proba': proba,
    }


# ---------------------------------------------------------------------------
# Evaluation and plots
# ---------------------------------------------------------------------------
def evaluate_and_plot(results_list, y_test, output_dir):
    class_names_list = [CLASS_NAMES[i] for i in range(N_CLASSES)]

    # ---------- per-model metrics ----------
    print("\n=== MODEL COMPARISON ===")
    print(f"{'Model':<25} {'Train':>8} {'Test':>8} {'Overfit':>9} {'F1 macro':>9}")
    print('-' * 62)
    for r in results_list:
        overfit = r['train_acc'] - r['test_acc']
        f1 = f1_score(y_test, r['preds'], average='macro')
        print(f"{r['name']:<25} {r['train_acc']:8.3f} {r['test_acc']:8.3f} "
              f"{overfit:9.3f} {f1:9.3f}")

    for r in results_list:
        print(f"\n{'=' * 50}")
        print(f"  {r['name']}")
        print(f"{'=' * 50}")
        print(classification_report(y_test, r['preds'],
                                    target_names=class_names_list))

    # ---------- confusion matrices ----------
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results_list):
        cm = confusion_matrix(y_test, r['preds'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names_list, yticklabels=class_names_list)
        ax.set_title(f"{r['name']}\nAccuracy: {r['test_acc']:.1%}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.suptitle('Confusion Matrices', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ---------- ROC curves (ice vs non-ice) ----------
    y_true_ice = np.isin(y_test, [0, 1]).astype(int)
    plt.figure(figsize=(8, 6))
    for r in results_list:
        if r['proba'] is not None:
            ice_proba = r['proba'][:, 0] + r['proba'][:, 1]
            fpr, tpr, _ = roc_curve(y_true_ice, ice_proba)
            roc_auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2.5,
                     label=f"{r['name']} (AUC = {roc_auc_val:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Comparison (Ice vs Non-Ice)', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ROC_Curve.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ---------- accuracy bar chart ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r['name'] for r in results_list]
    train_accs = [r['train_acc'] for r in results_list]
    test_accs = [r['test_acc'] for r in results_list]
    x = np.arange(len(names))
    w = 0.3
    ax.bar(x - w / 2, train_accs, w, label='Train', alpha=0.8)
    ax.bar(x + w / 2, test_accs, w, label='Test', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Accuracy')
    ax.set_title('Train vs Test Accuracy')
    ax.set_ylim(0.9, 1.01)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Model_Metrics.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Resolve paths relative to project root
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, args.data_dir)
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    X_train, y_train, X_test, y_test = load_data(data_dir)

    # Standardize features (for MLP and CNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler (needed at inference time)
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {output_dir}/scaler.pkl")

    # Train requested model(s)
    results_list = []

    if args.model in ('cnn', 'all'):
        print("\n--- Training CNN ---")
        r = train_cnn(X_train_scaled, y_train, X_test_scaled, y_test,
                       args.epochs, args.batch_size, output_dir)
        results_list.append(r)

    if args.model in ('mlp', 'all'):
        print("\n--- Training MLP architectures ---")
        r = train_mlp(X_train_scaled, y_train, X_test_scaled, y_test,
                       output_dir)
        results_list.append(r)

    if args.model in ('rf', 'all'):
        print("\n--- Training Random Forest ---")
        r = train_rf(X_train, y_train, X_test, y_test, output_dir)
        results_list.append(r)

    # Evaluate
    evaluate_and_plot(results_list, y_test, output_dir)

    # Summary
    best = max(results_list, key=lambda x: x['test_acc'])
    print(f"\nBest model: {best['name']} ({best['test_acc']:.1%} test accuracy)")


if __name__ == '__main__':
    main()
