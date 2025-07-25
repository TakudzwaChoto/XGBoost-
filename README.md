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
- [Comparison to Qualifying Round Standards](#comparison-to-qualifying-round-standards)  
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

This repository provides a comprehensive solution for predictive maintenance leveraging sensor data to:

- Predict equipment failure within a 7-day horizon (classification).  
- Estimate remaining useful life (RUL) for maintenance scheduling (regression).  

Two robust ensemble learning algorithms, Random Forest and XGBoost, are implemented and benchmarked for their effectiveness in this domain.

---

## Motivation

Unplanned equipment downtime results in significant operational costs, safety concerns, and productivity loss. Predictive maintenance addresses these challenges by forecasting failures and estimating RUL, enabling:

- Cost-effective and timely maintenance  
- Reduced downtime and improved asset reliability  
- Enhanced safety and operational efficiency  

---

## Dataset Description

| Aspect           | Description                                                      |
|------------------|------------------------------------------------------------------|
| **Records**      | > 400,000 sensor and operational readings                        |
| **Features**     | Sensor data (vibration, temperature, pressure, etc.) and operational metrics (scaled hours, machine types) |
| **Targets**      | - Binary failure indicator within 7 days (classification) <br> - Remaining Useful Life in hours (regression) |
| **Class Distribution** | Imbalanced (~6% failure cases)                               |

---

## Problem Statements

| Task                | Description                                  | Target Variable           | Type          |
|---------------------|----------------------------------------------|--------------------------|---------------|
| Failure Prediction   | Predict if equipment will fail within 7 days | `Failure_7Days` (0 or 1) | Classification |
| RUL Estimation      | Estimate remaining useful life in hours      | `Remaining_Useful_Life`   | Regression     |

---

## Methodology

### Data Preprocessing

- Median imputation for missing values.  
- Dropped features with no observed data to avoid imputation errors.  
- One-hot encoding of categorical variables (machine types).  
- Feature scaling applied to relevant numerical variables (`Operational_Hours_Scaled`).  
- Log transforms and robust clipping applied to reduce skewness and outliers.

### Feature Engineering

- Created interaction terms such as `Laser_Temp_Interaction`.  
- Included categorical machine types to capture equipment-specific behavior.  
- Derived additional features (e.g., `Temp_Vib_Ratio`, `Health_Index`) to better capture degradation.

### Modeling Techniques

| Algorithm       | Description                                        | Use Case                 |
|-----------------|--------------------------------------------------|--------------------------|
| Random Forest   | Ensemble of decision trees, reduces variance, interpretable | Regression & Classification |
| XGBoost         | Gradient boosting framework optimized for speed and accuracy | Regression & Classification |

---

## Model Training & Evaluation

### Regression Models: RUL Prediction

| Metric     | Random Forest | XGBoost  |
|------------|---------------|----------|
| MSE        | 2402.24       | 2417.12  |
| RMSE       | 49.01         | 49.16    |
| R² Score   | 0.9683        | 0.9681   |

*Both models demonstrate strong predictive accuracy for RUL.*

---

### Classification Models: Failure Prediction within 7 Days

| Metric    | Random Forest | XGBoost  |
|-----------|---------------|----------|
| Accuracy  | 93.5%         | 95.2%    |
| Precision | 48.4%         | 56.4%    |
| Recall    | 96.1%         | 87.9%    |
| F1 Score  | 64.4%         | 68.7%    |
| ROC AUC   | 97.7%         | 98.2%    |

*XGBoost shows higher precision and F1 score, while Random Forest offers better recall.*

---

### Cross-Validation (5-Fold)

| Model         | Task             | MSE ± Std Dev  | RMSE ± Std Dev  | R² ± Std Dev    |
|---------------|------------------|----------------|-----------------|-----------------|
| Random Forest | Regression (RUL) | 2408.79 ± 8.39 | 49.08 ± 0.09    | 0.9682 ± 0.0002 |
| XGBoost       | Regression (RUL) | 2428.49 ± 9.62 | 49.28 ± 0.10    | 0.9680 ± 0.0002 |

*Consistent performance across folds confirms model stability.*

---

## Feature Importance

| Rank | Feature                   | Importance (Random Forest) | Importance (XGBoost) |
|-------|---------------------------|----------------------------|---------------------|
| 1     | Operational_Hours_Scaled   | 96.61%                     | 85.82%              |
| 2     | Laser_Temp_Interaction     | 2.86%                      | 8.85%               |
| 3     | Vibration_mms              | 0.14%                      | —                   |
| 4     | Temperature_C              | 0.11%                      | —                   |
| 5     | Machine_Type_Valve_Controller | 0.05%                   | 0.55%               |
| 6     | Machine_Type_Vacuum_Packer | —                          | 0.62%               |

*Operational hours are the dominant predictor, with interaction terms contributing meaningfully.*

---

## Handling Missing Data

- Features without observed data (`Pressure_Flow_Ratio`, `Vibration_Increase_Rate`, `Temp_Increase_Rate`, `Health_Index`) were excluded during median imputation, preventing errors but potentially limiting model input.  
- Imputation warnings were systematically logged to aid transparency and future data collection improvements.

---

## Comprehensive Results Summary

| Model         | Task            | Metric    | Score    | Interpretation                      |
|---------------|-----------------|-----------|----------|-----------------------------------|
| Random Forest | RUL Regression  | RMSE      | 49.01    | High predictive accuracy           |
| Random Forest | Failure Class.  | Precision | 48.4%    | Moderate precision on failures    |
| Random Forest | Failure Class.  | Recall    | 96.1%    | Excellent detection of failures   |
| XGBoost       | RUL Regression  | RMSE      | 49.16    | Comparable accuracy to RF          |
| XGBoost       | Failure Class.  | Precision | 56.4%    | Higher precision vs. RF            |
| XGBoost       | Failure Class.  | Recall    | 87.9%    | Slightly lower recall than RF      |
| XGBoost       | Failure Class.  | ROC AUC   | 98.2%    | Excellent overall classification   |

---

## Comparison to Qualifying Round Standards

| Task                                 | Metric   | Achieved (XGBoost) | Qualifying Round Standard | Analysis                                                        |
|-------------------------------------|----------|--------------------|---------------------------|-----------------------------------------------------------------|
| **Task A: Failure Prediction (Binary Classification)** | Accuracy | 95.17%             | 78%                       | 22% higher accuracy indicating significantly better prediction. |
|                                     | Precision| 56.42%             | —                         | Better reliability in positive predictions.                     |
|                                     | Recall   | 87.91%             | —                         | Higher recall means fewer missed failures, critical for maintenance. |
|                                     | F1 Score | 68.73%             | —                         | Balanced and strong score for imbalanced data.                  |
|                                     | ROC AUC  | 0.9818             | —                         | Near-perfect class separation.                                  |
| **Task B: Remaining Useful Life (RUL) Prediction (Regression)** | MSE      | 2,417.12           | 19,298.90                 | 8× lower MSE, predictions much closer to true RUL.              |
|                                     | RMSE     | 49.01 days         | ~138.9 days               | Error margin significantly reduced, far more precise.           |
|                                     | R²       | 0.9681             | 0.7978                    | Explains 96.8% variance vs. 79.8%, indicating much better fit.  |

---

### Key Differentiators

| Aspect                | Proposed Approach               | Qualifying Round Approach         | Why It Matters                                      |
|-----------------------|--------------------------------|----------------------------------|----------------------------------------------------|
| Algorithm             | XGBoost + SMOTE                | Possibly basic logistic/linear regression | Handles non-linearity and imbalance better.          |
| Feature Engineering   | Advanced (e.g., `Temp_Vib_Ratio`, `Health_Index`) | Likely minimal (raw features only) | Captures degradation patterns more effectively.       |
| Class Imbalance Handling | SMOTE + class weights          | Possibly none                    | Higher recall by accounting for rare failure cases.  |
| Data Preprocessing    | Log transforms, scaling, clipping | May lack normalization          | Proper scaling aids convergence and generalization.  |
| Evaluation Rigor      | Cross-validation + threshold tuning | Simple train-test split         | Ensures robust, stable performance assessment.        |

#LANGUAGE: CHINESE
## Predictive Maintenance: Equipment Failure & Remaining Useful Life (RUL) Prediction  
### 预测性维护：设备故障与剩余寿命（RUL）预测
```
---
## 📋 Table of Contents  
## 📋 目录

- [Project Overview](#project-overview) / 项目概述  
- [Motivation](#motivation) / 研究动机  
- [Dataset Description](#dataset-description) / 数据集描述  
- [Problem Statements](#problem-statements) / 问题定义  
- [Methodology](#methodology) / 方法论  
  - [Data Preprocessing](#data-preprocessing) / 数据预处理  
  - [Feature Engineering](#feature-engineering) / 特征工程  
  - [Modeling Techniques](#modeling-techniques) / 建模技术  
- [Model Training & Evaluation](#model-training--evaluation) / 模型训练与评估  
  - [Regression Models](#regression-models) / 回归模型  
  - [Classification Models](#classification-models) / 分类模型  
  - [Cross-Validation](#cross-validation) / 交叉验证  
- [Feature Importance](#feature-importance) / 特征重要性  
- [Handling Missing Data](#handling-missing-data) / 缺失数据处理  
- [Comprehensive Results Summary](#comprehensive-results-summary) / 结果总结  
- [Project Structure](#project-structure) / 项目结构  
- [Installation Instructions](#installation-instructions) / 安装说明  
- [Usage Guide](#usage-guide) / 使用指南  
- [Dependencies](#dependencies) / 依赖环境  
- [Limitations & Known Issues](#limitations--known-issues) / 限制与已知问题  
- [Future Work & Enhancements](#future-work--enhancements) / 未来工作与改进  
- [Contributing Guidelines](#contributing-guidelines) / 贡献指南  
- [License](#license) / 许可协议  
- [Contact Information](#contact-information) / 联系方式

---

## Project Overview  
## 项目概述

This repository provides a comprehensive solution for **predictive maintenance** leveraging sensor data to:  
本项目通过传感器数据，提供全面的**预测性维护**解决方案，用于：

- **Predict equipment failure** within a 7-day horizon (classification).  
- **预测设备在未来7天内是否会发生故障（分类任务）**。  
- **Estimate remaining useful life (RUL)** for maintenance scheduling (regression).  
- **估计设备剩余使用寿命（RUL），辅助维护计划（回归任务）**。

Two robust ensemble learning algorithms, **Random Forest** and **XGBoost**, are implemented and benchmarked for their effectiveness in this domain.  
本项目实现并对比了两种强大的集成学习算法：**随机森林（Random Forest）**和**XGBoost**，验证其在本领域的表现。

---

## Motivation  
## 研究动机

Unplanned equipment downtime results in significant operational costs, safety concerns, and productivity loss. Predictive maintenance addresses these challenges by forecasting failures and estimating RUL, enabling:  
设备的非计划停机会带来高额运营成本、安全隐患和生产效率下降。预测性维护通过提前预警故障及估算剩余寿命，有效应对这些问题，实现：

- Cost-effective and timely maintenance  
- 成本节约且及时的维护安排  
- Reduced downtime and improved asset reliability  
- 降低停机时间，提升资产可靠性  
- Enhanced safety and operational efficiency  
- 增强安全性和运营效率  

---

## Dataset Description  
## 数据集描述

| Aspect / 内容           | Description / 说明                                    |
|------------------------|-----------------------------------------------------|
| **Records / 记录数**    | > 400,000 sensor and operational readings / 超过40万条传感器及操作数据 |
| **Features / 特征**     | Sensor data (vibration, temperature, pressure, etc.) and operational metrics (scaled hours, machine types) / 传感器数据（振动、温度、压力等）及操作指标（归一化运行时间，设备类型） |
| **Targets / 目标变量**  | - Binary failure indicator within 7 days (classification) <br> - Remaining Useful Life in hours (regression) / - 7天内设备故障二分类标签 <br> - 剩余使用寿命（小时）回归目标 |
| **Class Distribution / 类别分布** | Imbalanced (~6% failure cases) / 类别不平衡（约6%的故障样本）          |

---

## Problem Statements  
## 问题定义

| Task / 任务               | Description / 说明                             | Target Variable / 目标变量         | Type / 类型     |
|--------------------------|----------------------------------------------|----------------------------------|----------------|
| **Failure Prediction**    | Predict if equipment will fail within 7 days | `Failure_7Days` (0 or 1)          | Classification 分类  |
| **RUL Estimation**        | Estimate remaining useful life in hours      | `Remaining_Useful_Life`            | Regression 回归    |

---

## Methodology  
## 方法论

### Data Preprocessing  
### 数据预处理

- Median imputation for missing values.  
- 使用中位数填补缺失值。  
- Dropped features with no observed data to avoid imputation errors.  
- 删除无观测值特征以避免填补错误。  
- One-hot encoding of categorical variables (machine types).  
- 对类别变量（设备类型）进行独热编码。  
- Feature scaling applied to relevant numerical variables (`Operational_Hours_Scaled`).  
- 对数值型特征（归一化运行时间）进行特征缩放。

### Feature Engineering  
### 特征工程

- Created interaction terms such as `Laser_Temp_Interaction`.  
- 构造了激光温度交互特征等。  
- Included categorical machine types to capture equipment-specific behavior.  
- 引入设备类型类别特征以捕捉设备差异性。

### Modeling Techniques  
### 建模技术

| Algorithm / 算法          | Description / 说明                                  | Use Case / 适用场景               |
|--------------------------|--------------------------------------------------|--------------------------------|
| **Random Forest**         | Ensemble of decision trees, reduces variance, interpretable / 随机森林集成多棵决策树，降低方差，模型可解释 | Regression & Classification 回归与分类 |
| **XGBoost**               | Gradient boosting framework optimized for speed and accuracy / 高效准确的梯度提升框架    | Regression & Classification 回归与分类 |

---

## Model Training & Evaluation  
## 模型训练与评估

### Regression Models: RUL Prediction  
### 回归模型：剩余寿命预测

| Metric / 指标          | Random Forest      | XGBoost          |
|-----------------------|--------------------|------------------|
| **MSE**               | 2402.24            | 2417.12          |
| **RMSE**              | 49.01              | 49.16            |
| **R² Score**          | 0.9683             | 0.9681           |

*Both models demonstrate strong predictive accuracy for RUL.*  
*两种模型均展现了较高的剩余寿命预测准确度。*

---

### Classification Models: Failure Prediction within 7 Days  
### 分类模型：7天内故障预测

| Metric / 指标        | Random Forest      | XGBoost          |
|---------------------|--------------------|------------------|
| **Accuracy**        | 93.5%              | 95.2%            |
| **Precision**       | 48.4%              | 56.4%            |
| **Recall**          | 96.1%              | 87.9%            |
| **F1 Score**        | 64.4%              | 68.7%            |
| **ROC AUC**         | 97.7%              | 98.2%            |

*XGBoost shows higher precision and F1 score, while Random Forest offers better recall.*  
*XGBoost精确率和F1分数更高，随机森林召回率更佳。*

---

### Cross-Validation (5-Fold)  
### 交叉验证（5折）

| Model / 模型         | Task / 任务         | MSE ± Std Dev     | RMSE ± Std Dev    | R² ± Std Dev       |
|---------------------|---------------------|-------------------|-------------------|--------------------|
| Random Forest       | Regression (RUL)    | 2408.79 ± 8.39    | 49.08 ± 0.09      | 0.9682 ± 0.0002    |
| XGBoost             | Regression (RUL)    | 2428.49 ± 9.62    | 49.28 ± 0.10      | 0.9680 ± 0.0002    |

*Consistent performance across folds confirms model stability.*  
*稳定的交叉验证结果证明模型的鲁棒性。*

---
## Feature Importance  
## 特征重要性

| Rank / 排名 | Feature / 特征               | Importance (Random Forest) / 重要性 | Importance (XGBoost) / 重要性 |
|-------------|-----------------------------|-----------------------------------|------------------------------|
| 1           | Operational_Hours_Scaled     | 96.61%                            | 85.82%                       |
| 2           | Laser_Temp_Interaction       | 2.86%                             | 8.85%                        |
| 3           | Vibration_mms                | 0.14%                             | —                           |
| 4           | Temperature_C                | 0.11%                             | —                           |
| 5           | Machine_Type_Valve_Controller| 0.05%                             | 0.55%                        |
| 6           | Machine_Type_Vacuum_Packer   | —                                | 0.62%                        |

*Operational hours are the dominant predictor, with interaction terms contributing meaningfully.*  
*运行时间为最重要特征，交互项也有显著贡献。*
---
## Handling Missing Data  
## 缺失数据处理

- Features without observed data (`Pressure_Flow_Ratio`, `Vibration_Increase_Rate`, `Temp_Increase_Rate`, `Health_Index`) were **excluded** during median imputation, preventing errors but potentially limiting model input.  
- 对无观测值特征进行排除，避免填补错误，但可能影响模型输入完整性。  
- Imputation warnings were systematically logged to aid transparency and future data collection improvements.  
- 填补警告被系统记录，便于后续数据改进。

---
## Comprehensive Results Summary  
## 结果总结

| Model / 模型        | Task / 任务       | Metric / 指标    | Score / 得分    | Interpretation / 解释                      |
|--------------------|-------------------|------------------|-----------------|-------------------------------------------|
| Random Forest      | RUL Regression    | RMSE             | 49.01           | High predictive accuracy                   |
| Random Forest      | Failure Class.    | Precision        | 48.4%           | Moderate precision on failures            |
| Random Forest      | Failure Class.    | Recall           | 96.1%           | Excellent detection of failures            |
| XGBoost            | RUL Regression    | RMSE             | 49.16           | Comparable accuracy to RF                   |
| XGBoost            | Failure Class.    | Precision        | 56.4%           | Higher precision vs. RF                     |
| XGBoost            | Failure Class.    | Recall           | 87.9%           | Slightly lower recall than RF                |
| XGBoost            | Failure Class.    | ROC AUC          | 98.2%           | Excellent overall classification             |

---
## Project Structure  
## 项目结构

---
## Installation Instructions  
## 安装说明

### Prerequisites  
### 环境要求

- Python 3.7 or higher / Python 3.7及以上版本  
- pip package manager / pip包管理器

### Setup Steps  
### 安装步骤

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
python -m venv venv               # Optional: create virtual environment / 可选：创建虚拟环境
source venv/bin/activate          # Activate (Linux/macOS) / 激活环境（Linux/macOS）
# OR venv\Scripts\activate (Windows) / Windows环境激活
pip install -r requirements.txt  # Install dependencies / 安装依赖

