#%% md
##### Import necessary libraries and set seed for reproducibility
import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Set seeds for reproducibility
my_seed = 3
os.environ['PYTHONHASHSEED'] = '0'
random.seed(my_seed)
np.random.seed(my_seed)

#%% md
##### Load and preprocess dataset
# Load dataset
df = pd.read_csv(r"E:\桌面\活\活2（python）\特征选择-csv文件\continuous dataset.csv", index_col=0)
df.index = pd.to_datetime(df.index)

# Extract useful time features
df['hour_of_day'] = df.index.hour
df['month_of_year'] = df.index.month
df['day_of_week'] = df.index.isocalendar().day
df['working_day'] = df['day_of_week'].apply(lambda x: 0 if x > 5 else 1)

#%% md
##### RandomForest Feature Importance
# Define features and target
X = df.drop(columns="nat_demand").values
y = df['nat_demand'].values
feature_names = df.drop(columns="nat_demand").columns

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# Train RandomForest model
forest = RandomForestRegressor(max_depth=12, n_estimators=113, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)

# Extract feature importances
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

# Display feature importances
print("RandomForest Feature Importances:")
for i in range(len(feature_names)):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_names[indices], importances[indices])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('RandomForest Feature Importance')
plt.show()

#%% md
##### XGBoost Feature Importance with SHAP values
# Train XGBoost model
xgb_model = xgb.XGBRegressor(max_depth=4, learning_rate=0.1, n_estimators=100, objective='reg:squarederror', booster='gbtree')
xgb_model.fit(X_train, y_train)

# Use SHAP to explain feature importance
import shap
#%% Data Cleaning: Ensure X_train contains no null values and all numeric data
# Check for null values
print("Checking for null values in X_train:", np.any(pd.isnull(X_train)))

# Fill missing values with the median of the column
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# If there are missing values, fill them
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Check data types and convert to numeric if needed
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# Convert back to NumPy arrays after cleaning
X_train = X_train.values
X_test = X_test.values

#%% SHAP explanation
import shap

# Create SHAP explainer
explainer = shap.Explainer(xgb_model, X_train)

# Calculate SHAP values
shap_values = explainer(X_train)

# Plot SHAP summary
shap.summary_plot(shap_values, X_train, feature_names=df.columns[1:])

explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_train)

# Plot SHAP summary plot for feature importance
shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type="bar")

#%% md
##### Save Selected Features
selected_features = df[['nat_demand', 'T2M_toc', 'QV2M_toc', 'TQL_toc', 'W2M_toc', 'T2M_san',
                         'T2M_dav', 'Holiday_ID', 'holiday', 'hour_of_day', 'month_of_year',
                         'day_of_week', 'working_day']]

# Save final dataset with selected features
selected_features.to_csv(r"E:\桌面\活\活2（python）\特征选择-csv文件\筛选后的负荷与其它特征.csv", index=True, encoding='utf-8-sig')

# Save feature importances
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df.to_csv(r"E:\桌面\活\活2（python）\特征选择-csv文件\特征重要性.csv", index=False, encoding='utf-8-sig')
