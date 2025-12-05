// ============================================================================
// RUTOR GLACIER CLASSIFICATION 
// Earth Observation Advanced - Politecnico di Milano
// Classes: Ice, Water, Vegetation, Rock
// ============================================================================

Map.centerObject(rutorGlacier, 12);

// ============================================================================
// 1. PREPROCESSING
// ============================================================================

function processLandsat5(image) {
  var optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
            ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']);
  
  var thermal = image.select('ST_B6').multiply(0.00341802).add(149.0).rename('Thermal');
  
  var ndsi = optical.normalizedDifference(['Green', 'SWIR1']).rename('NDSI');
  var ndvi = optical.normalizedDifference(['NIR', 'Red']).rename('NDVI');
  var ndwi = optical.normalizedDifference(['Green', 'NIR']).rename('NDWI');
  
  return optical.addBands([thermal, ndsi, ndvi, ndwi])
    .updateMask(image.select('QA_PIXEL').bitwiseAnd(40).eq(0));
}

function processLandsat8(image) {
  var optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
            ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']);
  
  var thermal = image.select('ST_B10').multiply(0.00341802).add(149.0).rename('Thermal');
  
  var ndsi = optical.normalizedDifference(['Green', 'SWIR1']).rename('NDSI');
  var ndvi = optical.normalizedDifference(['NIR', 'Red']).rename('NDVI');
  var ndwi = optical.normalizedDifference(['Green', 'NIR']).rename('NDWI');
  
  return optical.addBands([thermal, ndsi, ndvi, ndwi])
    .updateMask(image.select('QA_PIXEL').bitwiseAnd(40).eq(0));
}

// ============================================================================
// 2. TRAINING DATA DEFINITION
// ============================================================================

var trainingCollection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(rutorGlacier)
  .filterDate('2020-08-01', '2024-09-30')
  .filter(ee.Filter.calendarRange(8, 9, 'month'))
  .filterMetadata('CLOUD_COVER', 'less_than', 20)
  .map(processLandsat8);

var trainingComposite = trainingCollection.median().clip(rutorGlacier);

var trainingPolygons = clean_ice.map(function(f) { return f.set('class', 0); })
  .merge(Debris_ice.map(function(f) { return f.set('class', 1); }))
  .merge(water_glacier.map(function(f) { return f.set('class', 2); }))
  .merge(vegetations.map(function(f) { return f.set('class', 3); }))
  .merge(mountains.map(function(f) { return f.set('class', 4); }));

var bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'Thermal', 'NDSI', 'NDVI', 'NDWI'];

var trainingData = trainingComposite.select(bands).sampleRegions({
  collection: trainingPolygons,
  properties: ['class'],
  scale: 30
});

// ============================================================================
// 3. SPLIT DATA: 75% TRAINING, 25% TESTING
// ============================================================================

// Add random column for splitting
var dataWithRandom = trainingData.randomColumn('random', 42);

// Split data: 75% for training, 25% for testing
var trainingSet = dataWithRandom.filter(ee.Filter.lt('random', 0.75));
var testingSet = dataWithRandom.filter(ee.Filter.gte('random', 0.75));

print('=== DATA SPLIT ===');
print('Total samples:', dataWithRandom.size());
print('Training samples (75%):', trainingSet.size());
print('Testing samples (25%):', testingSet.size());

// Check class distribution in both sets
print('Training class distribution:', trainingSet.aggregate_histogram('class'));
print('Testing class distribution:', testingSet.aggregate_histogram('class'));

// ============================================================================
// 4. TRAIN RANDOM FOREST CLASSIFIER
// ============================================================================

var classifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 200,
  seed: 42
}).train({
  features: trainingSet,  // Use only 75% for training
  classProperty: 'class',
  inputProperties: bands
});

// ============================================================================
// 4. TEMPORAL ANALYSIS - CLASSIFY PAST TIME PERIODS
// ============================================================================

// Define time periods for temporal analysis
var timePeriods = [
  {name: '1984-1988', start: '1984-08-01', end: '1988-09-30', sensor: 'L5'},
  {name: '1989-1993', start: '1989-08-01', end: '1993-09-30', sensor: 'L5'},
  {name: '1994-1998', start: '1994-08-01', end: '1998-09-30', sensor: 'L5'},
  {name: '1999-2003', start: '1999-08-01', end: '2003-09-30', sensor: 'L5'},
  {name: '2004-2008', start: '2004-08-01', end: '2008-09-30', sensor: 'L5'},
  {name: '2009-2013', start: '2009-08-01', end: '2013-09-30', sensor: 'L5'},
  {name: '2014-2018', start: '2014-08-01', end: '2018-09-30', sensor: 'L8'},
  {name: '2019-2024', start: '2019-08-01', end: '2024-09-30', sensor: 'L8'}
];

print('=== TEMPORAL ANALYSIS ===');
print('Creating composites for', timePeriods.length, 'time periods');

// Create composites for each time period
var temporalComposites = [];
var temporalClassifications = [];
var iceAreaData = [];

timePeriods.forEach(function(period) {
  
  var collection;
  if (period.sensor === 'L5') {
    collection = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
      .filterBounds(rutorGlacier)
      .filterDate(period.start, period.end)
      .filter(ee.Filter.calendarRange(8, 9, 'month'))
      .filterMetadata('CLOUD_COVER', 'less_than', 30)
      .map(processLandsat5);
  } else {
    collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
      .filterBounds(rutorGlacier)
      .filterDate(period.start, period.end)
      .filter(ee.Filter.calendarRange(8, 9, 'month'))
      .filterMetadata('CLOUD_COVER', 'less_than', 30)
      .map(processLandsat8);
  }
  
  // Check if we have data
  var imageCount = collection.size();
  print('Images available for', period.name + ':', imageCount);
  
  if (imageCount.getInfo() > 0) {
    // Create composite
    var composite = collection.median().clip(rutorGlacier);
    temporalComposites.push({
      name: period.name,
      composite: composite
    });
    
    // Classify the composite
    var classified = composite.select(bands).classify(classifier);
    temporalClassifications.push({
      name: period.name,
      classification: classified
    });
    
    // Calculate ice area (clean ice + debris ice)
    var totalIceArea = classified.lte(1)
      .multiply(ee.Image.pixelArea().divide(1e6))
      .reduceRegion({
        reducer: ee.Reducer.sum(),
        geometry: rutorGlacier,
        scale: 30,
        maxPixels: 1e9
      });
    
    var cleanIceArea = classified.eq(0)
      .multiply(ee.Image.pixelArea().divide(1e6))
      .reduceRegion({
        reducer: ee.Reducer.sum(),
        geometry: rutorGlacier,
        scale: 30,
        maxPixels: 1e9
      });
    
    var debrisIceArea = classified.eq(1)
      .multiply(ee.Image.pixelArea().divide(1e6))
      .reduceRegion({
        reducer: ee.Reducer.sum(),
        geometry: rutorGlacier,
        scale: 30,
        maxPixels: 1e9
      });
    
    iceAreaData.push(ee.Feature(null, {
      'period': period.name,
      'total_ice_km2': totalIceArea.get('classification'),
      'clean_ice_km2': cleanIceArea.get('classification'),
      'debris_ice_km2': debrisIceArea.get('classification')
    }));
    
  } else {
    print('No data available for period:', period.name);
  }
});

var iceEvolution = ee.FeatureCollection(iceAreaData);

print('=== TEMPORAL CLASSIFICATION COMPLETE ===');
print('Successfully processed', temporalComposites.length, 'time periods');

// ============================================================================
// 5. TESTING AND ACCURACY ASSESSMENT - SIMPLE VERSION
// ============================================================================

// Classify the testing set (25% of data)
var testingClassified = testingSet.classify(classifier);

// Calculate confusion matrix
var confusionMatrix = testingClassified.errorMatrix('class', 'classification');

print('=== ACCURACY RESULTS (25% TEST DATA) ===');
print('Overall Accuracy:', confusionMatrix.accuracy());
print('Kappa Coefficient:', confusionMatrix.kappa());
print('Confusion Matrix:', confusionMatrix);

// ============================================================================
// 6. CLASSIFICATION MAP + TEMPORAL VISUALIZATION
// ============================================================================

// Classify the current training composite
var rutorClassified = trainingComposite.select(bands).classify(classifier);

// ============================================================================
// 7. VISUALIZATION
// ============================================================================

var palette = ['#0066FF', '#4169E1', '#00FFFF', '#00FF00', '#A52A2A'];
var classNames = ['Clean Ice', 'Debris Ice', 'Water', 'Vegetation', 'Rock'];

// Add current classification
Map.addLayer(trainingComposite, {bands: ['Red', 'Green', 'Blue'], min: 0, max: 0.3}, 'Rutor Glacier (True Color)', false);
Map.addLayer(rutorClassified, {min: 0, max: 4, palette: palette}, 'Current Classification (2020-2024)');

// Add first and last temporal classifications if available
if (temporalClassifications.length > 0) {
  var firstTemporal = temporalClassifications[0];
  var lastTemporal = temporalClassifications[temporalClassifications.length - 1];
  
  Map.addLayer(firstTemporal.classification, {min: 0, max: 4, palette: palette}, 
               'First Period: ' + firstTemporal.name, false);
  Map.addLayer(lastTemporal.classification, {min: 0, max: 4, palette: palette}, 
               'Last Period: ' + lastTemporal.name, false);
  
  // Change detection
  var changeDetection = lastTemporal.classification.subtract(firstTemporal.classification);
  Map.addLayer(changeDetection, 
    {min: -4, max: 4, palette: ['red', 'orange', 'yellow', 'white', 'lightblue', 'blue', 'darkblue']}, 
    'Change: ' + firstTemporal.name + '→' + lastTemporal.name, false);
}

// Add training polygons
Map.addLayer(trainingPolygons, {color: 'red'}, 'Training Polygons', false);

// Legend
var legend = ui.Panel({style: {position: 'bottom-left', padding: '8px'}});
legend.add(ui.Label('5-Class Glacier Classification', {fontWeight: 'bold', fontSize: '16px'}));
legend.add(ui.Label('Temporal Analysis: 1984-2024', {fontSize: '12px', color: 'gray'}));

for (var i = 0; i < 5; i++) {
  var row = ui.Panel([
    ui.Label('', {backgroundColor: palette[i], padding: '8px', margin: '0'}),
    ui.Label(i + ': ' + classNames[i], {margin: '0 0 0 8px'})
  ], ui.Panel.Layout.Flow('horizontal'));
  legend.add(row);
}

Map.add(legend);

// ============================================================================
// 8. EXPORTS (CURRENT + TEMPORAL)
// ============================================================================

// Export current classification map
Export.image.toDrive({
  image: rutorClassified.toInt8(),
  description: 'Rutor_Current_Classification_2020_2024',
  folder: 'EarthObservation',
  region: rutorGlacier,
  scale: 30,
  maxPixels: 1e9
});

// Export temporal classifications
temporalClassifications.forEach(function(temporal) {
  var safeName = temporal.name.replace('-', '_');
  Export.image.toDrive({
    image: temporal.classification.toInt8(),
    description: 'Rutor_Classification_' + safeName,
    folder: 'EarthObservation',
    region: rutorGlacier,
    scale: 30,
    maxPixels: 1e9
  });
});

// Export ice evolution data (main output for plotting)
Export.table.toDrive({
  collection: iceEvolution,
  description: 'Rutor_Ice_Evolution_1984_2024',
  folder: 'EarthObservation',
  fileFormat: 'CSV'
});

// Export training data (75%)
Export.table.toDrive({
  collection: trainingSet,
  description: 'Training_Set_75_Percent',
  folder: 'EarthObservation',
  fileFormat: 'CSV'
});

// Export testing data and results (25%)
Export.table.toDrive({
  collection: testingSet,
  description: 'Testing_Set_25_Percent',
  folder: 'EarthObservation',
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: testingClassified,
  description: 'Testing_Results_25_Percent',
  folder: 'EarthObservation',
  fileFormat: 'CSV'
});

// Export simple accuracy summary
var accuracySummary = ee.FeatureCollection([
  ee.Feature(null, {
    'metric': 'Overall_Accuracy',
    'value': confusionMatrix.accuracy()
  }),
  ee.Feature(null, {
    'metric': 'Kappa_Coefficient',
    'value': confusionMatrix.kappa()
  }),
  ee.Feature(null, {
    'metric': 'Training_Samples',
    'value': trainingSet.size()
  }),
  ee.Feature(null, {
    'metric': 'Testing_Samples',
    'value': testingSet.size()
  }),
  ee.Feature(null, {
    'metric': 'Temporal_Periods_Processed',
    'value': temporalClassifications.length
  })
]);

Export.table.toDrive({
  collection: accuracySummary,
  description: 'Complete_Analysis_Summary',
  folder: 'EarthObservation',
  fileFormat: 'CSV'
});