# Predictive Maintenance: Equipment Failure & Remaining Useful Life (RUL) Prediction

---

## 📋 Table of Contents

- [Project Overview](#project-overview)  
- [Motivation](#motivation)  
- [Dataset Description](#dataset-description)  
- [Problem Statements](#problem-statements)  
- [Methodology](#methodology)  
  - [Data Preprocessing](#data-preprocessing)  
  - [Feature Engineering](#feature-engineering)  
  - [Modeling Techniques](#modeling-techniques)  
- [Model Training & Evaluation](#model-training--evaluation)  
  - [Regression Models](#regression-models)  
  - [Classification Models](#classification-models)  
  - [Cross-Validation](#cross-validation)  
- [Feature Importance](#feature-importance)  
- [Handling Missing Data](#handling-missing-data)  
- [Comprehensive Results Summary](#comprehensive-results-summary)  
- [Project Structure](#project-structure)  
- [Installation Instructions](#installation-instructions)  
- [Usage Guide](#usage-guide)  
- [Dependencies](#dependencies)  
- [Limitations & Known Issues](#limitations--known-issues)  
- [Future Work & Enhancements](#future-work--enhancements)  
- [Contributing Guidelines](#contributing-guidelines)  
- [License](#license)  
- [Contact Information](#contact-information)  

---

## Project Overview

This repository provides a comprehensive solution for **predictive maintenance** leveraging sensor data to:

- **Predict equipment failure** within a 7-day horizon (classification).  
- **Estimate remaining useful life (RUL)** for maintenance scheduling (regression).  

Two robust ensemble learning algorithms, **Random Forest** and **XGBoost**, are implemented and benchmarked for their effectiveness in this domain.

---

## Motivation

Unplanned equipment downtime results in significant operational costs, safety concerns, and productivity loss. Predictive maintenance addresses these challenges by forecasting failures and estimating RUL, enabling:

- Cost-effective and timely maintenance  
- Reduced downtime and improved asset reliability  
- Enhanced safety and operational efficiency  

---

## Dataset Description

| Aspect                  | Description                                         |
|------------------------|-----------------------------------------------------|
| **Records**             | > 400,000 sensor and operational readings          |
| **Features**            | Sensor data (vibration, temperature, pressure, etc.) and operational metrics (scaled hours, machine types) |
| **Targets**             | - Binary failure indicator within 7 days (classification) <br> - Remaining Useful Life in hours (regression) |
| **Class Distribution**  | Imbalanced (~6% failure cases)                       |

---

## Problem Statements

| Task                    | Description                                  | Target Variable           | Type          |
|-------------------------|----------------------------------------------|--------------------------|---------------|
| **Failure Prediction**    | Predict if equipment will fail within 7 days | `Failure_7Days` (0 or 1) | Classification |
| **RUL Estimation**        | Estimate remaining useful life in hours      | `Remaining_Useful_Life`   | Regression     |

---

## Methodology

### Data Preprocessing

- Median imputation for missing values.  
- Dropped features with no observed data to avoid imputation errors.  
- One-hot encoding of categorical variables (machine types).  
- Feature scaling applied to relevant numerical variables (`Operational_Hours_Scaled`).  

### Feature Engineering

- Created interaction terms such as `Laser_Temp_Interaction`.  
- Included categorical machine types to capture equipment-specific behavior.

### Modeling Techniques

| Algorithm        | Description                                                | Use Case                      |
|------------------|------------------------------------------------------------|-------------------------------|
| **Random Forest** | Ensemble of decision trees, reduces variance, interpretable | Regression & Classification    |
| **XGBoost**       | Gradient boosting framework optimized for speed and accuracy | Regression & Classification    |

---

## Model Training & Evaluation

### Regression Models: RUL Prediction

| Metric              | Random Forest      | XGBoost          |
|---------------------|--------------------|------------------|
| **MSE**             | 2402.24            | 2417.12          |
| **RMSE**            | 49.01              | 49.16            |
| **R² Score**        | 0.9683             | 0.9681           |

*Both models demonstrate strong predictive accuracy for RUL.*

---

### Classification Models: Failure Prediction within 7 Days

| Metric          | Random Forest      | XGBoost          |
|-----------------|--------------------|------------------|
| **Accuracy**    | 93.5%              | 95.2%            |
| **Precision**   | 48.4%              | 56.4%            |
| **Recall**      | 96.1%              | 87.9%            |
| **F1 Score**    | 64.4%              | 68.7%            |
| **ROC AUC**     | 97.7%              | 98.2%            |

*XGBoost shows higher precision and F1 score, while Random Forest offers better recall.*

---

### Cross-Validation (5-Fold)

| Model           | Task              | MSE ± Std Dev     | RMSE ± Std Dev    | R² ± Std Dev       |
|-----------------|-------------------|-------------------|-------------------|--------------------|
| Random Forest   | Regression (RUL)  | 2408.79 ± 8.39    | 49.08 ± 0.09      | 0.9682 ± 0.0002    |
| XGBoost         | Regression (RUL)  | 2428.49 ± 9.62    | 49.28 ± 0.10      | 0.9680 ± 0.0002    |

*Consistent performance across folds confirms model stability.*

---

## Feature Importance

| Rank | Feature                    | Importance (Random Forest) | Importance (XGBoost) |
|-------|----------------------------|----------------------------|---------------------|
| 1     | Operational_Hours_Scaled    | 96.61%                     | 85.82%              |
| 2     | Laser_Temp_Interaction      | 2.86%                      | 8.85%               |
| 3     | Vibration_mms               | 0.14%                      | —                   |
| 4     | Temperature_C               | 0.11%                      | —                   |
| 5     | Machine_Type_Valve_Controller | 0.05%                   | 0.55%               |
| 6     | Machine_Type_Vacuum_Packer  | —                          | 0.62%               |

*Operational hours are the dominant predictor, with interaction terms contributing meaningfully.*

---

## Handling Missing Data

- Features without observed data (`Pressure_Flow_Ratio`, `Vibration_Increase_Rate`, `Temp_Increase_Rate`, `Health_Index`) were **excluded** during median imputation, preventing errors but potentially limiting model input.  
- Imputation warnings were systematically logged to aid transparency and future data collection improvements.

---

## Comprehensive Results Summary

| Model           | Task            | Metric         | Score            | Interpretation                         |
|-----------------|-----------------|----------------|------------------|--------------------------------------|
| Random Forest   | RUL Regression  | RMSE           | 49.01            | High predictive accuracy              |
| Random Forest   | Failure Class.  | Precision      | 48.4%            | Moderate precision on failures       |
| Random Forest   | Failure Class.  | Recall         | 96.1%            | Excellent detection of failures       |
| XGBoost         | RUL Regression  | RMSE           | 49.16            | Comparable accuracy to RF              |
| XGBoost         | Failure Class.  | Precision      | 56.4%            | Higher precision vs. RF                |
| XGBoost         | Failure Class.  | Recall         | 87.9%            | Slightly lower recall than RF          |
| XGBoost         | Failure Class.  | ROC AUC        | 98.2%            | Excellent overall classification       |

---
## Project Structure
---
## Installation Instructions

### Prerequisites

- Python 3.7 or higher  
- pip package manager

### Setup Steps

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
python -m venv venv               # Optional: create virtual environment
source venv/bin/activate          # Activate (Linux/macOS)
# OR venv\Scripts\activate (Windows)
pip install -r requirements.txt  
