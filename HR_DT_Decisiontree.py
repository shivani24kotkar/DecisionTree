# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:51:23 2024

@author: Shivani
"""

import pandas as pd

# Load the dataset
hr_data = pd.read_csv('HR_DT.csv')

# Create a dataframe from the dataset
df = pd.DataFrame(hr_data)

# Rename the target column
df.rename(columns={'Sales': 'Target'}, inplace=True)

# Separate features (X) and target variable (y)
X = df.drop('Target', axis=1)
y = df['Target']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a Decision Tree Classifier model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the model
model_score = model.score(X_test, y_test)
print("Model Accuracy:", model_score)

# Predict using the trained model
y_predicted = model.predict(X_test)

# Compute confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print("Confusion Matrix:\n", cm)

# Visualize confusion matrix
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()