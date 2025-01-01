# Air and Water Quality Prediction

This repository contains a machine learning project that predicts the quality of air and water using datasets and various machine learning techniques. The models provide insights into air quality levels and water safety, aiming to assist in environmental monitoring and decision-making.

## Features
1. **Air Quality Prediction**:
   - Utilizes an SGD Regressor to predict Air Quality Index (AQI).
   - Classifies air quality into categories such as "Good", "Moderate", "Unhealthy", etc.

2. **Water Quality Prediction**:
   - Utilizes an AdaBoost Classifier to predict whether water is safe to drink.
   - Handles imbalanced datasets using RandomOverSampler.

## Project Workflow
### Air Quality Prediction
1. **Data Preprocessing**:
   - Handles missing values using KNN Imputer.
   - Standardizes features using StandardScaler.
   - Drops irrelevant columns such as Benzene.

2. **Model Training**:
   - Uses `SGDRegressor` for predicting AQI.
   - Provides air quality category based on AQI ranges.

### Water Quality Prediction
1. **Data Preprocessing**:
   - Cleans dataset by removing invalid entries.
   - Balances the dataset with oversampling techniques.

2. **Model Training**:
   - Uses a Decision Tree as the base estimator in an AdaBoost Classifier.
   - Outputs binary classifications (safe or unsafe).

## Air Quality Categories
| AQI Range   | Category                              |
|-------------|--------------------------------------|
| 0-50        | Good                                 |
| 51-100      | Moderate                             |
| 101-150     | Unhealthy for Sensitive Groups       |
| 151-200     | Unhealthy                            |
| 201-300     | Very Unhealthy                       |
| 301-500     | Hazardous                            |

## Results
- **Air Quality**: Predicts AQI and classifies into categories.
- **Water Quality**: Binary classification (safe/unsafe).
