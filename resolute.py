# Task 4
#Streamlit

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score

# Load datasets
@st.cache
def load_data(file_path):
    return pd.read_excel(file_path)

# Task 1: Clustering
def perform_clustering(train_data):
    train_data=train_data.drop('target')
    # Preprocess data: Encoding and scaling

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(train_data)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(data_scaled)
    
    train_data['Cluster'] = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    return kmeans, scaler, centroids, train_data

# Task 2: Classification
def train_classification_models(train_data, test_data):
    label_encoder = LabelEncoder()
    train_data['target'] = label_encoder.fit_transform(train_data['target'])
    
    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']
    X_test = test_data.copy()
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        train_accuracy = model.score(X_train, y_train)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        test_preds = model.predict(X_test)
        
        results[model_name] = {
            'train_accuracy': train_accuracy,
            'cv_mean_accuracy': cv_scores.mean(),
            'test_predictions': test_preds
        }
    
    return results

# Task 3: Data Analysis
def analyze_raw_data(data):
    data['time'] = data['time'].astype(str)
    data['date'] = data['date'].astype(str)
    
    
    data['timestamp'] = pd.to_datetime(data['date'] + ' ' + data['time'])
    data.sort_values(by='timestamp', inplace=True)
    data['duration'] = data['timestamp'].diff().dt.total_seconds().fillna(0)
    data['date'] = data['timestamp'].dt.date
    
    inside_data = data[data['position'] == 'inside']
    outside_data = data[data['position'] == 'outside']
    
    inside_duration = inside_data.groupby('date')['duration'].sum().reset_index(name='inside_duration')
    outside_duration = outside_data.groupby('date')['duration'].sum().reset_index(name='outside_duration')
    
    duration_df = pd.merge(inside_duration, outside_duration, on='date', how='outer').fillna(0)
    
    
    # picking_data = data[data['activity'] == 'picking']
    # placing_data = data[data['activity'] == 'placing']
    
    # picking_count = picking_data.groupby('date').size().reset_index(name='picking_count')
    # placing_count = placing_data.groupby('date').size().reset_index(name='placing_count')
    
    # activity_count_df = pd.merge(picking_count, placing_count, on='date', how='outer').fillna(0)

    activity_count_df = data.groupby(['date', 'activity']).size().unstack(fill_value=0).reset_index()
    
    final_df = pd.merge(duration_df, activity_count_df, on='date', how='outer').fillna(0)
    
    return final_df

# Streamlit App
st.title('Machine Learning and Data Analysis Dashboard')

# Upload datasets
raw_data_file = st.file_uploader("rawdata", type=["xlsx"])
train_data_file = st.file_uploader("train", type=["xlsx"])
test_data_file = st.file_uploader("test", type=["xlsx"])

if raw_data_file:
    raw_data = load_data(raw_data_file)
    st.header('Raw Data')
    st.write(raw_data.head())
    
    # Task 3: Data Analysis
    analysis_results = analyze_raw_data(raw_data)
    st.header('Date-wise Analysis of Inside/Outside Duration and Picking/Placing Activities')
    st.write(analysis_results)

if train_data_file and test_data_file:
    train_data = load_data(train_data_file)
    test_data = load_data(test_data_file)
    st.header('Train and Test Data')
    st.write('Train Data:', train_data.head())
    st.write('Test Data:', test_data.head())
    
    # Task 2: Classification
    classification_results = train_classification_models(train_data, test_data)
    for model_name, results in classification_results.items():
        st.header(f'{model_name} Results')
        st.write(f"Training Accuracy: {results['train_accuracy']}")
        st.write(f"Cross-Validation Mean Accuracy: {results['cv_mean_accuracy']}")
        st.write('Test Predictions:')
        st.write(results['test_predictions'])
        
if train_data_file and test_data_file:
    # Task 1: Clustering
    kmeans, scaler, centroids, clustered_data = perform_clustering(train_data.drop(columns=['target']))
    st.header('Clustering Results')
    st.write(clustered_data)
    
    st.header('Predict Cluster for a New Data Point')
    new_data_point = st.text_input('Enter new data point (comma separated values):')
    if new_data_point:
        new_data_point = np.array([float(x) for x in new_data_point.split(',')]).reshape(1, -1)
        new_data_point_scaled = scaler.transform(new_data_point)
        cluster = kmeans.predict(new_data_point_scaled)[0]
        distance = np.linalg.norm(new_data_point_scaled - centroids[cluster])
        st.write(f'The new data point belongs to cluster {cluster} with a distance of {distance} from the centroid.')
