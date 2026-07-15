This is the project :> 
# Dynamic Pricing for Ride-Sharing

Interactive Streamlit dashboard and analysis notebook for exploring ride prices and comparing fare-prediction models.

## Run the dashboard

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## Modelling approach

- The objective is to predict `Historical_Cost_of_Ride` from ride duration, rider demand, driver supply, customer information, location, booking time, and vehicle type.
- Numerical variables are converted to valid numeric types. For customers with no previous rides, the rating is replaced with the dataset mean so that a missing rating does not prevent a prediction.

- Feature engineering for time-based patterns (hour of day, day of week).

- Testing with gradient boosting models (XGBoost, LightGBM).

- Incorporating external factors like weather or traffic conditions.

- Implementing time-series cross-validation for more robust evaluation.

The objective is to predict Historical_Cost_of_Ride from ride duration, rider demand, driver supply, customer information, location, booking time, and vehicle type.

Numerical variables are converted to valid numeric types. For customers with no previous rides, the rating is replaced with the dataset mean so that a missing rating does not prevent a prediction.

- A Random Forest Regressor (n_estimators=100, max_depth=8) was trained on 70% of the data and evaluated on the remaining 30%. The model achieved an R² of 0.8469, with MAE = $55.52 and RMSE = $73.51, indicating strong predictive performance.

The model shows strong predictive performance, with ride duration being the dominant predictor. The exclusion of Cost_per_Minute from training ensures that the evaluation reflects a realistic, production-ready scenario where price is unknown at prediction time.