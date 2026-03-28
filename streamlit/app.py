import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset globally
wine = load_wine()

@st.cache_data
def prepare_model():
    data = pd.DataFrame(np.c_[wine['data'], wine['target']], columns=wine['feature_names'] + ['target'])
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)
    
    acc = accuracy_score(y_test, knn.predict(X_test_scaled))
    
    return scaler, knn, data, wine['target_names'], acc

scaler, knn, data, target_names, model_acc = prepare_model()

st.title("🍷 Wine Quality Prediction")
st.write(f"Model accuracy on test set: **{model_acc:.2%}** (matches notebook)")

# Sidebar for inputs
st.sidebar.header("Feature Inputs")
features = {}
for feature in wine.feature_names:
    features[feature] = st.sidebar.slider(
        feature.replace(' ', '_').title(),
        float(data[feature].min()),
        float(data[feature].max()),
        float(data[feature].mean()),
        0.01
    )

input_df = pd.DataFrame([features])

if st.button("Predict Quality", type="primary"):
    input_scaled = scaler.transform(input_df)
    prediction = knn.predict(input_scaled)[0]
    probabilities = knn.predict_proba(input_scaled)[0]
    
    st.success(f"Predicted Class: **{target_names[int(prediction)]}**")
    st.subheader("Probabilities:")
    probs_df = pd.DataFrame({
        'Class': target_names,
        'Probability': probabilities
    })
    st.dataframe(probs_df.style.format({'Probability': '{:.2%}'}))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Alcohol", f"{features['alcohol']:.2f}")
    with col2:
        st.metric("Predicted Quality", target_names[int(prediction)])

st.info("Matches notebook: KNN(n=3) on MinMax-scaled features only. PCA was viz-only.")
