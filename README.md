# Rutor Glacier temporal classification, comparing two different ML approaches

## Project Overview

Analysis of glacier retreat in the Italian Alps using Random Forest and Multi-Layer Perceptron classification of Landsat imagery. The study tracks ice loss, vegetation expansion, and environmental changes over 40 years.

## Key Results

- **99% accuracy** (Random Forest) | 98% accuracy MLP
- **50% ice loss** from 1984 to 2024
- **Vegetation increament and Water body creation** in exposed area
- **Cross-validation** on Monte-Rosa (96.9%)and Monte-Bianco (78.4%)

## Methodology

### Data & Classes
- **Satellite**: Landsat 5 and 8 (1984-2024, late summer)
- **Resolution**: 30m, 10 spectral bands (7 spectral band + 3 spectral indices(NDWI, NDVI, NDSI) )
- 
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
