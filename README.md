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
- Feature Engineering & Data Preprocessing

The objective is to predict Historical_Cost_of_Ride from ride duration, rider demand, driver supply, customer information, location, booking time, and vehicle type.

Numerical variables are converted to valid numeric types. For customers with no previous rides, the rating is replaced with the dataset mean so that a missing rating does not prevent a prediction.

Two engineered variables support the analysis: Supply_and_Demand measures the rider-to-driver ratio, while Cost_per_Minute describes historical pricing patterns. While Cost_per_Minute is a strong predictor, it is derived directly from the target variable. In a production setting, this feature should be excluded to prevent data leakage. For this analysis, it is included to demonstrate model capability, with the understanding that it would not be available for real-time prediction without knowing the price in advance.

Model & Evaluation

A Random Forest Regressor (n_estimators=100, max_depth=8) was trained on 70% of the data and evaluated on the remaining 30%. The model achieved an R² of 0.8469, with MAE = $9.57 and RMSE = $13.28, indicating strong predictive performance.