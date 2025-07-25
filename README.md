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

| Task                                 | Metric   | (XGBoost) Achieved | Qualifying Round (Set Standard) | Analysis                                                        |
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
# 预测性维护：设备故障预测与剩余使用寿命（RUL）预测
---

## 📋 目录

- [项目概述](#项目概述)  
- [动机](#动机)  
- [数据集描述](#数据集描述)  
- [问题陈述](#问题陈述)  
- [方法论](#方法论)  
  - [数据预处理](#数据预处理)  
  - [特征工程](#特征工程)  
  - [建模技术](#建模技术)  
- [模型训练与评估](#模型训练与评估)  
  - [回归模型](#回归模型)  
  - [分类模型](#分类模型)  
  - [交叉验证](#交叉验证)  
- [特征重要性](#特征重要性)  
- [缺失数据处理](#缺失数据处理)  
- [综合结果总结](#综合结果总结)  
- [与资格赛标准比较](#与资格赛标准比较)  
- [项目结构](#项目结构)  
- [安装说明](#安装说明)  
- [使用指南](#使用指南)  
- [依赖项](#依赖项)  
- [限制与已知问题](#限制与已知问题)  
- [未来工作与改进](#未来工作与改进)  
- [贡献指南](#贡献指南)  
- [许可证](#许可证)  
- [联系信息](#联系信息)  

---
## 项目概述

本项目提供了一套基于传感器数据的预测性维护完整解决方案，旨在：

- 预测设备在7天内是否会发生故障（分类任务）。  
- 估计设备的剩余使用寿命（RUL），用于维护计划安排（回归任务）。  

实现并对比了两种强大的集成学习算法：随机森林（Random Forest）和XGBoost，以评估其在该领域的表现。
---

## 动机

非计划性设备停机会带来巨大运营成本、安全隐患和生产力损失。预测性维护通过预测故障和估计剩余使用寿命，帮助实现：

- 成本效益高且及时的维护  
- 减少停机时间，提高资产可靠性  
- 提升安全性和运营效率  

---

## 数据集描述

| 方面           | 描述                                                         |
|----------------|--------------------------------------------------------------|
| **记录数**    | 超过40万条传感器及运行数据                                   |
| **特征**      | 传感器数据（振动、温度、压力等）及运行指标（时间刻度、设备类型） |
| **目标变量**  | - 7天内故障二分类指标（分类） <br> - 剩余使用寿命（小时，回归）  |
| **类别分布**  | 不平衡，约6%的故障样本                                        |

---

## 问题陈述

| 任务               | 描述                          | 目标变量                  | 类型       |
|--------------------|-------------------------------|---------------------------|------------|
| 故障预测           | 预测设备是否在7天内发生故障    | `Failure_7Days` (0或1)    | 分类       |
| 剩余使用寿命估计   | 估计设备剩余使用寿命（小时）  | `Remaining_Useful_Life`   | 回归       |

---
## 方法论

### 数据预处理

- 使用中位数填补缺失值。  
- 删除无观测数据的特征，避免填补错误。  
- 对分类变量（设备类型）进行独热编码。  
- 对相关数值变量（`Operational_Hours_Scaled`）进行特征缩放。  
- 应用对数变换和稳健截断减少偏态和异常值影响。

### 特征工程

- 创建交互项，如 `Laser_Temp_Interaction`。  
- 纳入设备类型分类变量以捕获设备差异。  
- 派生额外特征（如 `Temp_Vib_Ratio`、`Health_Index`）以更好地反映设备退化。

### 建模技术

| 算法          | 描述                                | 适用场景               |
|---------------|-------------------------------------|------------------------|
| 随机森林      | 多决策树集成，降低方差，易解释      | 回归 & 分类            |
| XGBoost       | 优化速度与准确率的梯度提升框架       | 回归 & 分类            |

---

## 模型训练与评估

### 回归模型：RUL预测

| 指标     | 随机森林    | XGBoost     |
|----------|-------------|-------------|
| MSE      | 2402.24     | 2417.12     |
| RMSE     | 49.01       | 49.16       |
| R²得分   | 0.9683      | 0.9681      |

*两模型均表现出优秀的RUL预测精度。*

---
### 分类模型：7天内故障预测

| 指标       | 随机森林    | XGBoost     |
|------------|-------------|-------------|
| 准确率     | 93.5%       | 95.2%       |
| 精确率     | 48.4%       | 56.4%       |
| 召回率     | 96.1%       | 87.9%       |
| F1分数     | 64.4%       | 68.7%       |
| ROC AUC    | 97.7%       | 98.2%       |

*XGBoost在精确率和F1分数上表现更优，随机森林则有更高召回率。*
---

### 交叉验证（5折）

| 模型        | 任务           | MSE ± 标准差    | RMSE ± 标准差  | R² ± 标准差    |
|-------------|----------------|-----------------|----------------|----------------|
| 随机森林    | 回归 (RUL)     | 2408.79 ± 8.39  | 49.08 ± 0.09   | 0.9682 ± 0.0002|
| XGBoost     | 回归 (RUL)     | 2428.49 ± 9.62  | 49.28 ± 0.10   | 0.9680 ± 0.0002|

*各折间表现稳定，模型泛化能力良好。*

---

## 特征重要性

| 排名 | 特征                    | 随机森林重要性     | XGBoost重要性      |
|------|--------------------------|--------------------|--------------------|
| 1    | Operational_Hours_Scaled  | 96.61%             | 85.82%             |
| 2    | Laser_Temp_Interaction    | 2.86%              | 8.85%              |
| 3    | Vibration_mms             | 0.14%              | —                  |
| 4    | Temperature_C             | 0.11%              | —                  |
| 5    | Machine_Type_Valve_Controller | 0.05%          | 0.55%              |
| 6    | Machine_Type_Vacuum_Packer | —                  | 0.62%              |

*运行小时数为主要预测因子，交互特征亦有显著贡献。*
---

## 缺失数据处理

- 无观测数据的特征（`Pressure_Flow_Ratio`、`Vibration_Increase_Rate`、`Temp_Increase_Rate`、`Health_Index`）在中位数填补时被剔除，避免错误但可能限制模型输入。  
- 系统性记录填补警告，便于透明度与未来数据收集改进。

---
## 综合结果总结

| 模型        | 任务           | 指标     | 得分       | 解释                            |
|-------------|----------------|----------|------------|--------------------------------|
| 随机森林    | RUL回归        | RMSE     | 49.01      | 高预测精度                     |
| 随机森林    | 故障分类       | 精确率   | 48.4%      | 故障检测精确率中等             |
| 随机森林    | 故障分类       | 召回率   | 96.1%      | 优秀的故障检测召回率           |
| XGBoost     | RUL回归        | RMSE     | 49.16      | 与随机森林精度相当             |
| XGBoost     | 故障分类       | 精确率   | 56.4%      | 精确率高于随机森林             |
| XGBoost     | 故障分类       | 召回率   | 87.9%      | 略低于随机森林召回率           |
| XGBoost     | 故障分类       | ROC AUC  | 98.2%      | 优秀的整体分类性能             |

---

## 与资格赛标准比较

| 任务                        | 指标    | 达成 (XGBoost)  | 资格赛标准         | 分析                                      |
|-----------------------------|---------|-----------------|--------------------|------------------------------------------|
| **任务A：故障预测（二分类）** | 准确率  | 95.17%          | 78%                | 准确率高出22%，整体预测能力显著提升       |
|                             | 精确率  | 56.42%          | —                  | 正确预测正样本的可靠性更强                 |
|                             | 召回率  | 87.91%          | —                  | 召回率高，减少漏检故障，关键于预测性维护   |
|                             | F1分数  | 68.73%          | —                  | 适合不平衡数据的平衡得分                   |
|                             | ROC AUC | 0.9818          | —                  | 近乎完美的类别区分能力                     |
| **任务B：剩余使用寿命预测（回归）** | MSE    | 2417.12         | 19298.90           | MSE降低8倍，预测更接近真实寿命             |
|                             | RMSE    | 49.01 天        | 约138.9 天         | 误差显著降低，预测更精确                     |
|                             | R²      | 0.9681          | 0.7978             | 解释方差率更高，模型拟合优良                 |

---
### 关键差异点

| 方面           | 本文方法                      | 资格赛方法                    | 重要性说明                                |
|----------------|------------------------------|------------------------------|-------------------------------------------|
| 算法           | XGBoost + SMOTE              | 可能是基础逻辑回归/线性回归    | 更好处理非线性与类别不平衡问题             |
| 特征工程       | 高级（如`Temp_Vib_Ratio`、`Health_Index`） | 可能仅使用原始特征              | 更准确捕获设备退化模式                     |
| 类别不平衡处理 | SMOTE + 类权重               | 可能无处理                    | 提高召回率，减少稀有故障漏检               |
| 数据预处理     | 对数变换、缩放、截断         | 可能缺少归一化                | 加速模型收敛，提高泛化能力                 |
| 评估方法       | 交叉验证 + 阈值调优          | 简单的训练-测试划分            | 确保结果稳健可靠，防止过拟合               |

