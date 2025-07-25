# Predictive Maintenance: Equipment Failure & Remaining Useful Life (RUL) Prediction

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Motivation](#motivation)  
- [Dataset Description](#dataset-description)  
- [Problem Statements](#problem-statements)  
- [Methodology](#methodology)  
  - [Data Preprocessing](#data-preprocessing)  
  - [Feature Engineering](#feature-engineering)  
  - [Modeling Approaches](#modeling-approaches)  
- [Model Training and Evaluation](#model-training-and-evaluation)  
  - [Random Forest Regression](#random-forest-regression)  
  - [XGBoost Regression](#xgboost-regression)  
  - [Random Forest Classification](#random-forest-classification)  
  - [XGBoost Classification](#xgboost-classification)  
  - [Cross-Validation](#cross-validation)  
- [Feature Importance Analysis](#feature-importance-analysis)  
- [Handling Missing Data](#handling-missing-data)  
- [Results Summary](#results-summary)  
- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Dependencies](#dependencies)  
- [Known Issues and Limitations](#known-issues-and-limitations)  
- [Future Work](#future-work)  
- [Contributing](#contributing)  
- [License](#license)  
- [Contact](#contact)  

---

## Project Overview

This project addresses the predictive maintenance challenges faced in industrial equipment monitoring, specifically:

- **Failure Prediction:** Predict whether equipment will fail within a defined future timeframe, allowing proactive maintenance.  
- **Remaining Useful Life (RUL) Estimation:** Predict the time left before equipment failure to optimize maintenance scheduling.

By leveraging sensor data and operational metrics, we build and benchmark two powerful ensemble machine learning models — Random Forest and XGBoost — for both regression and classification tasks.

---

## Motivation

Unexpected equipment failures lead to costly downtime, safety hazards, and operational inefficiencies. Predictive maintenance aims to forecast failures and estimate remaining equipment lifespan, enabling:

- Reduction of unplanned downtime  
- Cost savings on maintenance  
- Improved equipment reliability and lifespan  
- Enhanced safety and operational efficiency  

This project provides a practical machine learning pipeline to facilitate predictive maintenance in manufacturing and industrial environments.

---

## Dataset Description

The dataset contains over 400,000 records with the following characteristics:

- **Features:**  
  - Sensor readings such as vibration (mms), temperature (C), pressure-flow ratios, and their interaction terms  
  - Operational parameters like scaled operational hours  
  - Categorical features for machine types (e.g., Valve Controller, Vacuum Packer)  
- **Target Variables:**  
  - **Failure within 7 days (binary classification):** 1 if failure occurs within next 7 days, else 0  
  - **Remaining Useful Life (RUL) (regression):** Numeric value estimating remaining hours of operation  

The dataset is imbalanced for failure prediction (around 6% failures).

---

## Problem Statements

- **Task A (Classification):** Predict whether the equipment will fail within the next 7 days.  
- **Task B (Regression):** Predict the remaining useful life (RUL) of the equipment.

---

## Methodology

### Data Preprocessing

- Handling missing values using median imputation.  
- Encoding categorical variables.  
- Scaling continuous features where appropriate (e.g., operational hours).  
- Detecting and removing features with no observed data to avoid imputation errors.

### Feature Engineering

- Creating interaction terms such as `Laser_Temp_Interaction`.  
- Including machine type as categorical variables with one-hot encoding.

### Modeling Approaches

- **Random Forest:** An ensemble of decision trees providing robustness to noise and interpretability.  
- **XGBoost:** Gradient boosting framework known for high performance and speed.

Both models are trained and evaluated on classification and regression tasks.

---

## Model Training and Evaluation

### Random Forest Regression

- **Performance:**  
  - Mean Squared Error (MSE): ~2402.24  
  - Root Mean Squared Error (RMSE): ~49.01  
  - R² Score: ~0.9683  
- Trained to predict Remaining Useful Life (RUL).  
- Feature importance shows `Operational_Hours_Scaled` as dominant.

### XGBoost Regression

- **Performance:**  
  - MSE: ~2417.12  
  - RMSE: ~49.16  
  - R² Score: ~0.9681  
- Similar performance to Random Forest, slightly different feature importance distribution.

### Random Forest Classification

- Predicts failure within 7 days.  
- Evaluated with accuracy, precision, recall, F1-score, and ROC AUC.  
- Robust to imbalanced classes due to ensemble bagging.

### XGBoost Classification

- Often better handling of imbalanced data through parameter tuning.  
- Slightly better precision and ROC AUC in reported results.

### Cross-Validation

- 5-fold cross-validation used for robust performance estimates.  
- Warnings logged for features without observed values in imputation step, which are skipped.  
- Cross-validation confirms model stability and consistency.

---

## Feature Importance Analysis

Top features across models:

| Rank | Feature                    | Importance (Random Forest) | Importance (XGBoost) |
|-------|----------------------------|----------------------------|---------------------|
| 1     | Operational_Hours_Scaled    | 0.9661                     | 0.8582              |
| 2     | Laser_Temp_Interaction      | 0.0286                     | 0.0885              |
| 3     | Vibration_mms               | 0.0014                     | —                   |
| 4     | Temperature_C               | 0.0011                     | —                   |
| 5     | Machine_Type_Valve_Controller | 0.0005                  | 0.0055              |
| 6     | Machine_Type_Vacuum_Packer  | —                          | 0.0062              |

---

## Handling Missing Data

- Features without any observed values (`Pressure_Flow_Ratio`, `Vibration_Increase_Rate`, `Temp_Increase_Rate`, `Health_Index`) were skipped during median imputation.  
- Warnings are generated during data preprocessing to inform about skipped features.  
- Imputation strategy ensures model robustness despite missingness but suggests data collection improvements.

---

## Results Summary

| Model             | Task                 | Metric        | Score         |
|-------------------|----------------------|---------------|---------------|
| Random Forest     | Regression (RUL)     | MSE           | 2402.24       |
| Random Forest     | Regression (RUL)     | RMSE          | 49.01         |
| Random Forest     | Regression (RUL)     | R²            | 0.9683        |
| Random Forest     | Classification       | Accuracy      | ~93.5%        |
| Random Forest     | Classification       | Precision     | ~48.4%        |
| Random Forest     | Classification       | Recall        | ~96.1%        |
| Random Forest     | Classification       | F1 Score      | ~64.4%        |
| Random Forest     | Classification       | ROC AUC       | ~97.7%        |
| XGBoost           | Regression (RUL)     | MSE           | 2417.12       |
| XGBoost           | Regression (RUL)     | RMSE          | 49.16         |
| XGBoost           | Regression (RUL)     | R²            | 0.9681        |
| XGBoost           | Classification       | Accuracy      | ~95.2%        |
| XGBoost           | Classification       | Precision     | ~56.4%        |
| XGBoost           | Classification       | Recall        | ~87.9%        |
| XGBoost           | Classification       | F1 Score      | ~68.7%        |
| XGBoost           | Classification       | ROC AUC       | ~98.2%        |

---

## Project Structure


---

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### Train and evaluate models

```bash
python train_models.py
