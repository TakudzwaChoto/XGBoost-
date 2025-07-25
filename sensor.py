# ======================
# COMPLETE SENSOR DATA ANALYSIS SOLUTION
# ======================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (accuracy_score, recall_score, f1_score, 
                           precision_score, confusion_matrix, roc_auc_score, 
                           mean_squared_error, r2_score)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
from tqdm import tqdm

# Configure settings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style('whitegrid')

# ============================================
# 1. DATA LOADING
# ============================================

def load_data(filepath):
    """Load and clean the dataset"""
    dtype_spec = {
        'Machine_ID': 'category',
        'Machine_Type': 'category',
        'Installation_Year': 'int16',
        'Operational_Hours': 'float32',
        'Temperature_C': 'float32',
        'Vibration_mms': 'float32',
        'Sound_dB': 'float32',
        'Oil_Level_pct': 'float32',
        'Coolant_Level_pct': 'float32',
        'Power_Consumption_kW': 'float32',
        'Last_Maintenance_Days_Ago': 'int16',
        'Maintenance_History_Count': 'int16',
        'Failure_History_Count': 'int16',
        'AI_Supervision': 'bool',
        'Error_Codes_Last_30_Days': 'string',
        'Remaining_Useful_Life_days': 'int16',
        'Failure_Within_7_Days': 'int8',
        'Laser_Intensity': 'float32',
        'Hydraulic_Pressure_bar': 'float32',
        'Coolant_Flow_L_min': 'float32',
        'Heat_Index': 'float32',
        'AI_Override_Events': 'int16'
    }
    
    print("\nLoading data...")
    try:
        df = pd.read_csv(filepath, dtype=dtype_spec)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Clean infinite values
    for col in df.select_dtypes(include=['float']):
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    print(f"\nData loaded successfully. Shape: {df.shape}")
    print("\nClass balance:")
    print(df['Failure_Within_7_Days'].value_counts(normalize=True))
    return df

# ============================================
# 2. ENHANCED FEATURE ENGINEERING
# ============================================

def preprocess_data(df, for_task='A'):
    """Create features with competition-specific enhancements"""
    # Basic features
    df['Error_Count'] = df['Error_Codes_Last_30_Days'].str.count(';').fillna(0).astype('int8')
    
    # Maintenance features
    df['Maintenance_Urgency'] = np.where(
        df['Maintenance_History_Count'] > 0,
        np.log1p(df['Last_Maintenance_Days_Ago'] / df['Maintenance_History_Count']),
        0
    )
    
    # Sensor interactions
    df['Temp_Vib_Ratio'] = (df['Temperature_C'].clip(1) / df['Vibration_mms'].clip(1)).replace([np.inf, -np.inf], 10)
    df['Power_Stress'] = (df['Power_Consumption_kW'] * df['Vibration_mms']).clip(upper=1000)
    df['Pressure_Flow_Ratio'] = (df['Hydraulic_Pressure_bar'] / df['Coolant_Flow_L_min'].clip(0.1)).replace([np.inf, -np.inf], 10)
    df['Laser_Temp_Interaction'] = df['Laser_Intensity'] * df['Temperature_C']
    
    # Operational features
    df['Operational_Hours_Scaled'] = np.log1p(df['Operational_Hours'])
    df['AI_Intervention_Rate'] = df['AI_Override_Events'] / (df['Operational_Hours'] + 1)
    
    # Convert bool to int
    df['AI_Supervision'] = df['AI_Supervision'].astype('int8')
    
    # Task B specific features
    if for_task == 'B':
        # Degradation rate features
        df['Vibration_Increase_Rate'] = df.groupby('Machine_ID')['Vibration_mms'].transform(
            lambda x: x.diff().rolling(30, min_periods=1).mean()
        )
        df['Temp_Increase_Rate'] = df.groupby('Machine_ID')['Temperature_C'].transform(
            lambda x: x.diff().rolling(30, min_periods=1).mean()
        )
        # Health index (composite metric)
        df['Health_Index'] = (
            0.4 * (1 - df['Vibration_Increase_Rate']) + 
            0.3 * (1 - df['Temp_Increase_Rate']) + 
            0.2 * (df['Oil_Level_pct'] / 100) + 
            0.1 * (df['Coolant_Level_pct'] / 100)
        )
    
    return df

# ============================================
# 3. MODEL PIPELINES (TASK A & B)
# ============================================

def build_classification_model(use_xgboost=True):
    """Build pipeline for Task A (Classification)"""
    numeric_features = [
        'Operational_Hours_Scaled', 'Temperature_C', 'Vibration_mms',
        'Sound_dB', 'Oil_Level_pct', 'Coolant_Level_pct',
        'Power_Consumption_kW', 'Maintenance_Urgency',
        'Failure_History_Count', 'Error_Count',
        'Temp_Vib_Ratio', 'Power_Stress',
        'Laser_Intensity', 'Hydraulic_Pressure_bar',
        'Coolant_Flow_L_min', 'Heat_Index',
        'Pressure_Flow_Ratio', 'Laser_Temp_Interaction',
        'AI_Intervention_Rate'
    ]
    
    categorical_features = ['Machine_Type', 'AI_Supervision']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    if use_xgboost:
        classifier = XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=200,
            class_weight={0:1, 1:3},
            max_depth=10,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
    
    return make_pipeline(
        preprocessor,
        SMOTE(sampling_strategy=0.25, random_state=42, k_neighbors=5),
        classifier
    )

def build_regression_model(use_xgboost=True):
    """Build pipeline for Task B (Regression)"""
    numeric_features = [
        'Operational_Hours_Scaled', 'Temperature_C', 'Vibration_mms',
        'Sound_dB', 'Oil_Level_pct', 'Coolant_Level_pct',
        'Power_Consumption_kW', 'Maintenance_Urgency',
        'Failure_History_Count', 'Error_Count',
        'Temp_Vib_Ratio', 'Power_Stress',
        'Laser_Intensity', 'Hydraulic_Pressure_bar',
        'Coolant_Flow_L_min', 'Heat_Index',
        'Pressure_Flow_Ratio', 'Laser_Temp_Interaction',
        'AI_Intervention_Rate',
        'Vibration_Increase_Rate', 'Temp_Increase_Rate',
        'Health_Index'
    ]
    
    categorical_features = ['Machine_Type', 'AI_Supervision']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    if use_xgboost:
        regressor = XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror'
        )
    else:
        regressor = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            max_features=0.7,
            random_state=42,
            n_jobs=-1
        )
    
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])

# ============================================
# 4. EVALUATION FUNCTIONS
# ============================================

def evaluate_classification_model(model, X_test, y_test, model_name):
    """Evaluation for Task A (Classification)"""
    print("\n" + "="*50)
    print(f"{model_name.upper()} CLASSIFICATION PERFORMANCE")
    print("="*50)
    
    with tqdm(total=4, desc="Evaluating") as pbar:
        y_pred = model.predict(X_test)
        pbar.update(1)
        
        try:
            y_proba = model.predict_proba(X_test)[:,1]
        except:
            y_proba = model.decision_function(X_test)
        pbar.update(1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        pbar.update(1)
        
        # Print results
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix')
        plt.show()
        pbar.update(1)

def evaluate_regression_model(model, X_test, y_test, model_name):
    """Evaluation for Task B (Regression)"""
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"{model_name.upper()} REGRESSION PERFORMANCE")
    print("="*50)
    print(f"\nMSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Feature importance
    if hasattr(model.named_steps['regressor'], 'feature_importances_'):
        preprocessor = model.named_steps['preprocessor']
        
        # Get feature names
        numeric_features = [
            'Operational_Hours_Scaled', 'Temperature_C', 'Vibration_mms',
            'Sound_dB', 'Oil_Level_pct', 'Coolant_Level_pct',
            'Power_Consumption_kW', 'Maintenance_Urgency',
            'Failure_History_Count', 'Error_Count',
            'Temp_Vib_Ratio', 'Power_Stress',
            'Laser_Intensity', 'Hydraulic_Pressure_bar',
            'Coolant_Flow_L_min', 'Heat_Index',
            'Pressure_Flow_Ratio', 'Laser_Temp_Interaction',
            'AI_Intervention_Rate',
            'Vibration_Increase_Rate', 'Temp_Increase_Rate',
            'Health_Index'
        ]
        
        categorical_features = ['Machine_Type', 'AI_Supervision']
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        categorical_names = ohe.get_feature_names_out(categorical_features)
        all_features = numeric_features + list(categorical_names)
        
        importances = model.named_steps['regressor'].feature_importances_
        
        # Handle potential length mismatch
        min_length = min(len(all_features), len(importances))
        feature_importances = pd.DataFrame({
            'feature': all_features[:min_length],
            'importance': importances[:min_length]
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 Most Important Features:")
        for i, row in feature_importances.head(5).iterrows():
            print(f"{i+1}. {row['feature']}: {row['importance']:.4f}")
        
        # Visualization
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', 
                    data=feature_importances.head(15))
        plt.title(f'Top 15 Predictive Features ({model_name})')
        plt.tight_layout()
        plt.show()
    
    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Residual Plot ({model_name})')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.show()

# ============================================
# 5. CROSS-VALIDATION FUNCTIONS
# ============================================

def run_classification_cv(model, X, y, model_name):
    """CV for Task A (Classification)"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    print(f"\nRunning 5-fold CV for {model_name}...")
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    print("\nCross-Validation Results:")
    print(f"Accuracy: {np.mean(scores['test_accuracy']):.4f} ± {np.std(scores['test_accuracy']):.4f}")
    print(f"Precision: {np.mean(scores['test_precision']):.4f} ± {np.std(scores['test_precision']):.4f}")
    print(f"Recall: {np.mean(scores['test_recall']):.4f} ± {np.std(scores['test_recall']):.4f}")
    print(f"F1: {np.mean(scores['test_f1']):.4f} ± {np.std(scores['test_f1']):.4f}")
    print(f"ROC AUC: {np.mean(scores['test_roc_auc']):.4f} ± {np.std(scores['test_roc_auc']):.4f}")

def run_regression_cv(model, X, y, model_name):
    """CV for Task B (Regression)"""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'mse': 'neg_mean_squared_error',
        'rmse': 'neg_root_mean_squared_error', 
        'r2': 'r2'
    }
    
    print(f"\nRunning 5-fold CV for {model_name}...")
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    print("\nCross-Validation Results:")
    print(f"MSE: {-np.mean(scores['test_mse']):.2f} ± {np.std(scores['test_mse']):.2f}")
    print(f"RMSE: {-np.mean(scores['test_rmse']):.2f} ± {np.std(scores['test_rmse']):.2f}")
    print(f"R²: {np.mean(scores['test_r2']):.4f} ± {np.std(scores['test_r2']):.4f}")

# ============================================
# 6. TASK SOLUTIONS
# ============================================

def task_a_solution(df):
    """Solution for Task A (Failure Prediction)"""
    print("\nPreprocessing data for Task A...")
    df = preprocess_data(df, for_task='A')
    
    features = [
        'Machine_Type', 'Operational_Hours_Scaled', 'Temperature_C',
        'Vibration_mms', 'Sound_dB', 'Oil_Level_pct',
        'Coolant_Level_pct', 'Power_Consumption_kW',
        'Maintenance_Urgency', 'Failure_History_Count',
        'Error_Count', 'AI_Supervision',
        'Temp_Vib_Ratio', 'Power_Stress',
        'Laser_Intensity', 'Hydraulic_Pressure_bar',
        'Coolant_Flow_L_min', 'Heat_Index',
        'Pressure_Flow_Ratio', 'Laser_Temp_Interaction',
        'AI_Intervention_Rate'
    ]
    target = 'Failure_Within_7_Days'
    
    X = df[features]
    y = df[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Random Forest
    print("\nTraining Random Forest model...")
    rf_model = build_classification_model(use_xgboost=False)
    with tqdm(total=100, desc="Random Forest") as pbar:
        rf_model.fit(X_train, y_train)
        pbar.update(100)
    evaluate_classification_model(rf_model, X_test, y_test, "Random Forest")
    run_classification_cv(rf_model, X_train, y_train, "Random Forest")
    
    # XGBoost
    print("\nTraining XGBoost model...")
    xgb_model = build_classification_model(use_xgboost=True)
    with tqdm(total=100, desc="XGBoost") as pbar:
        xgb_model.fit(X_train, y_train)
        pbar.update(100)
    evaluate_classification_model(xgb_model, X_test, y_test, "XGBoost")
    run_classification_cv(xgb_model, X_train, y_train, "XGBoost")

def task_b_solution(df):
    """Solution for Task B (RUL Prediction)"""
    print("\nPreprocessing data for Task B...")
    df = preprocess_data(df, for_task='B')
    
    features = [
        'Machine_Type', 'Operational_Hours_Scaled', 'Temperature_C',
        'Vibration_mms', 'Sound_dB', 'Oil_Level_pct',
        'Coolant_Level_pct', 'Power_Consumption_kW',
        'Maintenance_Urgency', 'Failure_History_Count',
        'Error_Count', 'AI_Supervision',
        'Temp_Vib_Ratio', 'Power_Stress',
        'Laser_Intensity', 'Hydraulic_Pressure_bar',
        'Coolant_Flow_L_min', 'Heat_Index',
        'Pressure_Flow_Ratio', 'Laser_Temp_Interaction',
        'AI_Intervention_Rate',
        'Vibration_Increase_Rate', 'Temp_Increase_Rate',
        'Health_Index'
    ]
    target = 'Remaining_Useful_Life_days'
    
    # Filter out machines with RUL <= 0 (already failed)
    df = df[df[target] > 0]
    X = df[features]
    y = df[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Random Forest
    print("\nTraining Random Forest model...")
    rf_model = build_regression_model(use_xgboost=False)
    with tqdm(total=100, desc="Random Forest") as pbar:
        rf_model.fit(X_train, y_train)
        pbar.update(100)
    evaluate_regression_model(rf_model, X_test, y_test, "Random Forest")
    run_regression_cv(rf_model, X_train, y_train, "Random Forest")
    
    # XGBoost
    print("\nTraining XGBoost model...")
    xgb_model = build_regression_model(use_xgboost=True)
    with tqdm(total=100, desc="XGBoost") as pbar:
        xgb_model.fit(X_train, y_train)
        pbar.update(100)
    evaluate_regression_model(xgb_model, X_test, y_test, "XGBoost")
    run_regression_cv(xgb_model, X_train, y_train, "XGBoost")

# ============================================
# 7. MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    try:
        # Load data
        df = load_data(r"C:\Users\ZYNC\Desktop\数据分析：train_data.csv")
        if df is None:
            raise Exception("Failed to load data")
        
        # Run Task A
        print("\n" + "="*50)
        print("STARTING TASK A: FAILURE PREDICTION")
        print("="*50)
        task_a_solution(df.copy())
        
        # Clear memory
        print("\nClearing memory for Task B...")
        del df
        gc.collect()
        
        # Reload data for Task B
        df = load_data(r"C:\Users\ZYNC\Desktop\数据分析：train_data.csv")
        if df is None:
            raise Exception("Failed to reload data for Task B")
        
        # Run Task B
        print("\n" + "="*50)
        print("STARTING TASK B: REMAINING USEFUL LIFE PREDICTION")
        print("="*50)
        task_b_solution(df.copy())
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        print("\nExecution completed.")