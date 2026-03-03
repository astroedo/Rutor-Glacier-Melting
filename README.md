# Rutor Glacier Temporal Classification: A Comparative ML/DL Approach

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-JavaScript-green.svg)](https://earthengine.google.com/)

## Project Overview
This repository contains the codebase for analyzing the retreat of the Rutor Glacier (Italian Alps) using satellite imagery. The study compares classical Machine Learning (Random Forest) and Deep Learning (Multi-Layer Perceptron, 1D Convolutional Neural Networks) approaches to classify temporal changes over a 40-year period (1984–2024). 

The primary goals are:
1. Track ice loss and quantify the retreat.
2. Monitor ecological changes, such as vegetation expansion and the formation of new water bodies.
3. Compare the feature-extraction capabilities of 1D-CNNs on spectral signatures against traditional pixel-based classifiers.

## Repository Structure

    ├── data/                   # Training and testing datasets (CSV)
    │   ├── Training_Set_75_Percent.csv
    │   └── Testing_Set_25_Percent.csv
    ├── docs/                   # Project reports and documentation
    ├── presentation/           # Slides and presentation materials
    ├── src/                    # Source code
    │   ├── data_collection/    # Google Earth Engine (GEE) scripts for data extraction
    │   └── models/             # Jupyter Notebooks for model training and evaluation (CNN, MLP)
    ├── Result/                 # Output plots (ROC curves, metrics)
    ├── requirements.txt        # Python dependencies
    └── README.md

## Dataset & Features
- **Satellite Data**: Landsat 5 (TM) and 8 (OLI/TIRS), late summer acquisitions to minimize snow cover.
- **Location**: Rutor Glacier, Graian Alps (45.67°N, 6.98°E).
- **Features**: 30m spatial resolution. Each pixel is represented by a 10-dimensional spectral signature:
  - 7 Spectral Bands (Visible, NIR, SWIR, Thermal)
  - 3 Normalized Indices: `NDSI` (Snow), `NDVI` (Vegetation), `NDWI` (Water)
- **Target Classes**: `(0)` Clean ice, `(1)` Debris-covered ice, `(2)` Water, `(3)` Vegetation, `(4)` Rock.

## Methodology & Models

### 1. 1D-CNN (Deep Learning)
Treats the 10-dimensional spectral signature as a sequence to extract local correlations between adjacent spectral bands.
- **Architecture**: 2x Conv1D layers (64 filters, kernel size 3) + Batch Normalization + MaxPooling + Dense Head.
- **Optimizer**: Adam with learning rate scheduling (`ReduceLROnPlateau`).

### 2. Multi-Layer Perceptron (Deep Learning)
Evaluated across different depths to study the capacity required for spectral classification.
- **Architecture**: Triple-Layer (128, 64, 32), Double-Layer, and Single-Layer variants.

### 3. Random Forest (Machine Learning Baseline)
Implemented directly in Google Earth Engine for scalable, cloud-native inference.
- **Hyperparameters**: 110 Trees, Gini impurity.

## Key Results

| Model Approach | Test Accuracy | Overfitting Margin |
|:---|:---:|:---:|
| **Random Forest (GEE)** | 99.1% | 0.9% |
| **1D-CNN (TensorFlow)** | 98.4% | 1.4% |
| **MLP (Triple-Layer)** | 98.4% | 0.9% |
| **MLP (Double-Layer)** | 97.5% | 1.8% |
| **MLP (Single-Layer)**| 96.9% | 0.7% |

The temporal classification analysis over the 40-year study period unequivocally demonstrates that the Rutor Glacier is experiencing severe and rapid melting. The total ice area has drastically reduced, losing approximately 50% of its coverage since 1984.

*Developed as part of MSc Geoinformatics Engineering - Earth Observation Advanced course, Politecnico di Milano*
