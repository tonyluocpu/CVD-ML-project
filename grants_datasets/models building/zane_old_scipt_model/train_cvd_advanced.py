# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from alibi.explainers import IntegratedGradients, CounterfactualProto
from sklearn.inspection import permutation_importance
import json



# === Load and Merge Data ===
# Load datasets
df_main = pd.read_csv('final.csv')
df_main['statecounty'] = df_main['statecounty'].astype(str).apply(lambda x: x.split()[0])

df_target = pd.read_csv('cvd_mortality_2018_2020.csv', index_col=0)
df_target['statecounty'] = df_target['statecounty'].apply(lambda x: x.split()[0])

# Remove duplicates in target data
df_target = df_target.drop_duplicates(subset='statecounty', keep='first')

# Impute missing values in target columns
allRaceCVD = df_target.columns[2]
APICVD = df_target.columns[1]
df_target[allRaceCVD] = df_target[allRaceCVD].fillna(df_target[APICVD])

# Merge datasets
df = pd.merge(df_main, df_target, on='statecounty', how='inner')
print(f"Number of rows after merging: {len(df)}")

# Identify islands (specific condition, optional)
islands = df['statecounty'].apply(lambda x: x.startswith('15'))
islandnames = df[islands]['statecounty']

# Remove statecounty column after processing
df = df.drop(columns=['statecounty'])

# === Select Relevant Columns ===
# Select columns containing 'Per' or 'per' in their names
columns_to_keep = [
    col for col in df.columns if ('Per' in col or 'per' in col) and 'Upper' not in col and 'Housing' not in col
]
columns_to_keep.append('NHPI_Single_Race_Civilian_Population_With_Health_Insurance')
target = 'allrace_cvd_mortality_per_100k_2018_2020'

# Filter dataset
df = df[columns_to_keep]

# === Data Cleaning ===
# Replace infinite values and outliers
large_value_threshold = 1e308
df = df.replace([np.inf, -np.inf], np.nan)
df = df.applymap(lambda x: np.nan if isinstance(x, float) and abs(x) > large_value_threshold else x)

# Drop rows where all columns are NaN
df.dropna(how='all', inplace=True)

# Separate target variable
y = df[target]
X = df.drop(columns=[target])

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric columns: {len(numeric_cols)}, Categorical columns: {len(categorical_cols)}")

# === Preprocessing Pipelines ===
# Numeric preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine preprocessors
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# === Apply Preprocessing ===
# Transform features
X_processed = preprocessor.fit_transform(X)

# Validate processed data
assert not np.any(np.isnan(X_processed)), "Processed data contains NaN values."
assert not np.any(np.isinf(X_processed)), "Processed data contains infinity values."

# Impute missing values in the target variable
y = y.replace([np.inf, -np.inf], np.nan)
y = pd.to_numeric(y, errors='coerce')

imputer_y = SimpleImputer(strategy='mean')
y_imputed = pd.Series(imputer_y.fit_transform(y.values.reshape(-1, 1)).flatten(), name=target)

# === Split Data ===
X_train, X_val, y_train, y_val = train_test_split(X_processed, y_imputed, test_size=0.2, random_state=42)

# === Advanced model training and tuning ===

# Import necessary libraries
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import json

# === Model Training and Evaluation ===

# Define and initialize models
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
catboost_model = CatBoostRegressor(verbose=0, random_state=42)

# Stacking ensemble for improved performance
stacked_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('catboost', catboost_model)
    ],
    final_estimator=XGBRegressor(random_state=42)
)

# Train the models
print("Training XGBoost model...")
xgb_model.fit(X_train, y_train)

print("Training CatBoost model...")
catboost_model.fit(X_train, y_train)
print("Training Stacked model...")
stacked_model.fit(X_train, y_train)

# === Model Evaluation ===
models = {
    'XGBoost': xgb_model,
    'CatBoost': catboost_model,
    'Stacked': stacked_model
}

# Evaluate all models on the validation set
results = {}
for name, model in models.items():
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    explained_var = explained_variance_score(y_val, y_pred)
    results[name] = {'MSE': mse, 'R2': r2, 'Explained Variance': explained_var}
    print(f"{name} Results: MSE={mse:.4f}, R2={r2:.4f}, Explained Variance={explained_var:.4f}")

    # Save predictions for external analysis
    np.savetxt(f'{name}_predictions.csv', np.c_[y_val, y_pred], delimiter=',', header='True,Predicted', comments='')

# Save evaluation metrics to a JSON file
with open('model_results.json', 'w') as f:
    json.dump(results, f)

# === Explainability with Feature Importances ===
from sklearn.inspection import permutation_importance

print("Calculating feature importances for Stacked model...")
perm_importances = permutation_importance(stacked_model, X_val, y_val, n_repeats=10, random_state=42)
feature_importances = {
    feature: importance for feature, importance in zip(preprocessor.get_feature_names_out(), perm_importances.importances_mean)
}

# Save feature importances for analysis
with open('feature_importances.json', 'w') as f:
    json.dump(feature_importances, f)

# === Summary ===
print("\nModel Evaluation Summary:")
for name, metrics in results.items():
    print(f"{name} - MSE: {metrics['MSE']:.4f}, R2: {metrics['R2']:.4f}, Explained Variance: {metrics['Explained Variance']:.4f}")
