This is the project :> 
# Dynamic Pricing for Ride-Sharing

Interactive Streamlit dashboard and analysis notebook for exploring ride prices and comparing fare-prediction models.

## Run the dashboard

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## Modelling approach

- `Historical_Cost_of_Ride` is the target.
- `Cost_per_Minute` is used only for EDA. It is excluded from training because it is calculated from the target and would leak the answer into the model.
- Linear Regression, Decision Tree, and Random Forest are compared with the same shuffled five-fold cross-validation splits.
- The model with the lowest mean CV RMSE is selected, then reported on a separate 20% test split.

Shared preparation and modelling code lives in `model_utils.py`, so the dashboard and notebook use the same logic.
