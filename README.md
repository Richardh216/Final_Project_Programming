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
- Two engineered variables support the analysis: `Supply_and_Demand` measures the rider-to-driver ratio, while `Cost_per_Minute` describes historical pricing patterns. Because `Cost_per_Minute` is calculated from the target price, it is used only for EDA and is excluded from model training to prevent data leakage.
