import ee

ee.Authenticate()
ee.Initialize(project="crop-yield-prediction-509513")

print(ee.String("Earth Engine connected successfully").getInfo())