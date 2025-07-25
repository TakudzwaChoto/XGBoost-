# Industrial Equipment Predictive Maintenance System
       colsample_bytree=0.8,
       scale_pos_weight=3,
       random_state=42,
       eval_metric='logloss'
   )
## Task B: RUL Estimation
**Objective:** Regression to predict remaining useful life (days)
**Approach:**
1. **Feature engineering:**
   - Degradation rate calculations:
     ```math
     Vibration\_Increase\_Rate = \frac{\Delta Vibration}{\Delta Time}
     ```
   - Health index composite metric:
     ```math
     Health\_Index = 0.4(1-VibRate) + 0.3(1-TempRate) + 0.2(OilLevel) + 0.1(CoolantLevel)
     ```
   - Rolling window statistics (30-day windows)
2. **Evaluation metrics:**
   - MSE: ```math \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2 ```
   - RMSE: ```math \sqrt{MSE} ```
   - R²: ```math 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2} ```
**Algorithms:**
1. **XGBoost Regressor:**
   ```python
   XGBRegressor(
       n_estimators=500,
       max_depth=8,
       learning_rate=0.05,
       subsample=0.8,
       colsample_bytree=0.8,
       random_state=42,
       objective='reg:squarederror'
   )
## Performance Results
### Task A: Classification Performance
| Model          | Accuracy | Precision | Recall | F1    | ROC AUC |
|----------------|----------|-----------|--------|-------|---------|
| Random Forest  | 0.9358   | 0.4843    | 0.9607 | 0.6440| 0.9773  |
| XGBoost        | 0.9517   | 0.5642    | 0.8791 | 0.6873| 0.9818  |
### Task B: Regression Performance
| Model          | MSE     | RMSE  | R²     |
|----------------|---------|-------|--------|
| Random Forest  | 2402.24 | 49.01 | 0.9683 |
| XGBoost        | 2417.12 | 49.16 | 0.9681 |
## Getting Started
### Installation
```bash
git clone https://github.com/yourusername/predictive-maintenance.git
cd predictive-maintenance
python -m pip install -r requirements.txt
