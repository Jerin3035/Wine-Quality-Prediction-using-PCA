# Wine Quality Prediction using PCA & KNN

## Overview
This repository contains a machine learning project for predicting wine quality using the [sklearn wine dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset).

### Notebook
- **Wine_Quality_Prediction.ipynb**: EDA, preprocessing (MinMaxScaler), PCA visualization, KNN model training (97% accuracy on scaled features).

### Streamlit Deployment
Deployed interactive app:
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://wine-quality-prediction-using-pca-eenfxjbhbras5vuuhpuedg.streamlit.app/)

- **app.py**: Streamlit UI for feature inputs, predictions with probabilities.
- **requirements.txt**: Dependencies.

## Local Setup & Run
```bash
git clone https://github.com/Jerin3035/Wine-Quality-Prediction-using-PCA.git
cd Wine-Quality-Prediction-using-PCA
pip install -r requirements.txt
streamlit run app.py
```

## Model Details
- Dataset: 13 features, 3 classes.
- Preprocessing: MinMaxScaler.
- Model: KNeighborsClassifier(n_neighbors=3).
- Accuracy: ~97%.

See notebook for full analysis.
