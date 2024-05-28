# Project Overview
This project comprises four tasks, focusing on different aspects of machine learning and data analysis. The tasks include clustering, classification, data aggregation, and building a simple web application using Streamlit to represent the results.

# List of contents:
1. Tasks
2. Required libraries
3. 

# Tasks
Task 1: Machine Learning - Clustering
Objective: Use a clustering technique to extract patterns or segregate data into groups.

User Story: Users should be able to provide a data point (a row) and the program should identify the cluster to which the given data point belongs and explain why.

Hint: Use any clustering algorithm such as K-means, DBSCAN, or hierarchical clustering.

Task 2: Machine Learning - Classification
Objective: Train classification model(s) on the training dataset and test the algorithm on the test dataset. Share the predicted target values for evaluation.

Train multiple classification algorithms (e.g., logistic regression, random forests, SVM).

Share the target values for each algorithm.

Provide the training accuracy and explain the choice of algorithms.

Task 3: Python Data Aggregation
Objective: Use the provided raw data to derive specific metrics.

Calculate the total duration for each "inside" and "outside" activity on a date-wise basis.

Calculate the date-wise number of "picking" and "placing" activities.

Sample output is provided in the output sheet.

Task 4: Basic Streamlit
Task: Represent output of above 3 tasks either in streamlit.

# Required Libraries
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, cross_val_score
