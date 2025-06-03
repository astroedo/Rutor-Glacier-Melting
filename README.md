# Rutor Glacier ML Monitoring Project

## Advanced Machine Learning for Alpine Glacier Change Detection Using Earth Observation Data

### Project Overview

This project applies cutting-edge machine learning techniques to monitor the Rutor Glacier in the Italian Alps using multi-temporal Landsat satellite imagery from Google Earth Engine. The study compares traditional Random Forest ensemble methods with modern Convolutional Neural Networks (CNN) for glacier classification and change detection over the period 2013-2023.

### Key Objectives

- **Develop advanced classification algorithms** to distinguish glacier features (clean ice, debris-covered ice, snow, rock, water, vegetation)
- **Compare ML approaches**: Traditional Random Forest vs. modern CNN for spectral pattern recognition
- **Quantify glacier retreat** during the last decade using late-summer imagery when snow coverage is minimal
- **Create operational framework** for automated glacier monitoring using Earth Observation data

### Technical Innovation

- **Custom 1D CNN architecture** adapted for multispectral Landsat data analysis
- **Hybrid feature engineering** combining spectral indices (NDSI, NDVI, NDWI) with thermal and topographic data
- **Advanced validation framework** with temporal cross-validation and uncertainty quantification
- **Google Earth Engine integration** with local Python ML pipeline for scalable processing

### Study Area

**Rutor Glacier, Graian Alps** (45.67°N, 6.98°E)
- Elevation range: 2,400-3,400m
- Area: ~8 km² (historically)
- Located on Italy-France border
- Significant retreat documented in recent decades

### Methodology Highlights

1. **Data Collection**: Multi-temporal Landsat 8-9 Surface Reflectance (2013-2023)
2. **Preprocessing**: Cloud masking, atmospheric correction, spectral index calculation
3. **Classification**: Comparative analysis of Random Forest vs CNN performance
4. **Change Detection**: Post-classification comparison and statistical trend analysis
5. **Validation**: Cross-validation with high-resolution imagery and literature data

### Expected Outcomes

- **High-accuracy glacier classification** maps with uncertainty bounds
- **Quantitative retreat analysis** including area loss rates and terminus position changes
- **Algorithm performance comparison** demonstrating CNN advantages for complex spectral patterns
- **Reproducible methodology** applicable to other Alpine glaciers for climate change monitoring

### Technical Stack

- **Google Earth Engine**: Cloud-based satellite data processing
- **Python**: Machine learning pipeline (TensorFlow, scikit-learn, pandas)
- **GitHub**: Version control and project management
- **Jupyter Notebooks**: Interactive analysis and visualization

### Academic Contribution

This project demonstrates the application of advanced machine learning techniques to real-world environmental monitoring challenges, bridging traditional remote sensing methods with modern deep learning approaches for climate change research.

---

*Developed as part of MSc Geoinformatics Engineering - Earth Observation Advanced course, Politecnico di Milano*
