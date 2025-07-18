# Rutor Glacier ML Monitoring Project

## Advanced Machine Learning for Alpine Glacier Change Detection Using Earth Observation Data

### Project Overview

This project analyzes the temporal evolution of Rutor Glacier in the Italian Alps over a 40-year period (1984-2024) using satellite remote sensing and machine learning classification techniques. The study demonstrates significant glacier retreat and environmental changes using state-of-the-art Earth Observation methodologies.

### Key Objectives

- **Temporal Classification:** Track glacier changes across 8 time periods (5-year intervals)
- **Machine Learning Comparison:** Evaluate Random Forest vs Multi-Layer Perceptron performance
- **Climate Impact Assessment:** Quantify ice loss and environmental transformation
- **Methodology Validation** Test model transferability across different Alpine glaciers

### Technical Innovation

- **Overall Accuracy:** 99% (Random Forest) | 98% (MLP)
- **Ice Loss:** Significant retreat observed from 1984-2024
- **Environmental Changes:** Increased vegetation and water areas
- **Model Performance:** RF outperformed MLP for glacier classification
- **Cross-Validation:** 96.9% accuracy on Monte Rosa, 78.4% on Mont Blanc

### Study Area

**Rutor Glacier, Graian Alps** (45.67°N, 6.98°E)
- Elevation range: 2,400-3,400m
- Area: ~8 km² (historically)
- Located on Italy-France border
- Significant retreat documented in recent decades

### Methodology Highlights

1. **Data Collection**: Multi-spectral Landsat 5 TM & Landsat 8 OLI/TIRS
  1.1 Temporal Coverage: August 1984 - September 2024
  1.2 Spatial resolution: 30m
2. **Preprocessing**
  2.1 Cloud masking
  2.2 spectral index calculation
   
         // Cloud masking and spectral indices calculation
        var ndsi = optical.normalizedDifference(['Green', 'SWIR1']).rename('NDSI');
        var ndvi = optical.normalizedDifference(['NIR', 'Red']).rename('NDVI');
        var ndwi = optical.normalizedDifference(['Green', 'NIR']).rename('NDWI');
3. **Classification**:
     (0)- Clean ice
     (1)- Debris-covered ice
     (2)- Water
     (3)- Vegetation
     (4)- Rock
4. **Machine learning implementation**
   Random Forest (Google earth engine)

       ''' javascript
       var classifier = ee.Classifier.smileRandomForest({
        numberOfTrees: 110,
        seed: 42
       }).train({
          features: trainingSet,
          classProperty: 'class',
          inputProperties: bands
       });
   Multi-Layer Perceptron (Google Colab)

       '''python 
       mlp_classifier = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=1000,
        random_state=42
       )

### Key results

- **Randome Forest accuracy**: 99.1%, Overfitting 0.9%
- **MLP Single-Layer accuracy**: 96.9%, Overfitting 0.7%
- **MLP Double-Layer accuracy**: 97.5%, Overfitting 1.8%
- **MLP Triple-Layer accuracy**: 98.4%, Overfitting 0.9%
  
### Feature importance

- NDWI: 27%
- NDVI: 18%
- Thermal: 9%

### Temporal changes

- **Total ice lost**: ~50%
- New water bodies formed 

---

*Developed as part of MSc Geoinformatics Engineering - Earth Observation Advanced course, Politecnico di Milano*
