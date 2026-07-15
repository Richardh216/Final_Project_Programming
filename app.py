import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Dynamic Pricing Dashboard", layout="wide", page_icon="🚗")

@st.cache_resource
def initial_setup():
    df = pd.read_csv('dynamic_pricing.csv')
    
    cols_to_convert = ['Average_Ratings', 'Historical_Cost_of_Ride', 'Expected_Ride_Duration', 'Number_of_Riders', 'Number_of_Drivers']
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    mean_rating = df['Average_Ratings'].mean()
    df.loc[df['Number_of_Past_Rides'] == 0, 'Average_Ratings'] = mean_rating
    
    #New Columns
    df['Cost_per_Minute'] = df['Historical_Cost_of_Ride'] / df['Expected_Ride_Duration']
    df['Supply_and_Demand'] = df['Number_of_Riders'] / df['Number_of_Drivers'].replace(0, 1)
    
    categorical_cols = ['Vehicle_Type', 'Location_Category', 'Time_of_Booking', 'Customer_Loyalty_Status']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    X = df_encoded.drop(columns=['Historical_Cost_of_Ride', 'Cost_per_Minute'])
    y = df_encoded['Historical_Cost_of_Ride']
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return df, model, feature_names, X_test, y_test, y_pred

df, model, feature_names, X_test, y_test, y_pred = initial_setup()

# --- SIDEBAR - LIVE SIMULATOR ---
st.sidebar.header("🚗 Ride Simulator Parameters")

duration = st.sidebar.slider("Expected Ride Duration (minutes)", 5, 180, 25)
riders = st.sidebar.slider("Number of Riders (Demand)", 1, 100, 50)
drivers = st.sidebar.slider("Number of Available Drivers (Supply)", 0, 100, 25)

location = st.sidebar.selectbox("Location Category", ["Urban", "Suburban", "Rural"])
time_of_day = st.sidebar.selectbox("Time of Booking", ["Morning", "Afternoon", "Evening", "Night"])
vehicle = st.sidebar.selectbox("Vehicle Type", ["Economy", "Premium"])
loyalty = st.sidebar.selectbox("Customer Loyalty Status", ["Regular", "Silver", "Gold"])
past_rides = st.sidebar.number_input("Number of Past Rides", min_value=0, value=15)
rating = st.sidebar.slider("Customer Rating", 1.0, 5.0, 4.5)

# Calculation of simulated supply-demand ratio
simulated_ratio = riders / (drivers if drivers > 0 else 1)

# --- MAIN PAGE ---
st.title("📊 Ride-Sharing Dynamic Pricing Dashboard")
st.markdown("Interactive tool for exploration and real-time algorithmic price prediction.")

tab1, tab2, tab3 = st.tabs(["💰 Price Predictor", "📈 Business Insights", "⚙️ Model Diagnostics"])

# --- TAB 1: PRICE PREDICTION ---
with tab1:
    st.subheader("Real-Time Algorithmic Fare Estimation")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Simulated Demand/Supply Ratio", value=f"{simulated_ratio:.2f}")
        if simulated_ratio > 1.5:
            st.warning("⚠️ High demand spike detected! Surge pricing multipliers are active.")
        elif simulated_ratio < 0.6:
            st.success("✅ High driver availability. Standard base rates applied.")
            
    with col2:
        # Calculation of prediction after clicking the button
        if st.button("Calculate Estimated Fare", type="primary"):
            input_data = pd.DataFrame(0, index=[0], columns=feature_names)
            
            input_data['Expected_Ride_Duration'] = duration
            input_data['Number_of_Riders'] = riders
            input_data['Number_of_Drivers'] = drivers
            input_data['Number_of_Past_Rides'] = past_rides
            input_data['Average_Ratings'] = rating
            input_data['Supply_and_Demand'] = simulated_ratio
            
            if f"Vehicle_Type_{vehicle}" in feature_names: input_data[f"Vehicle_Type_{vehicle}"] = 1
            if f"Location_Category_{location}" in feature_names: input_data[f"Location_Category_{location}"] = 1
            if f"Time_of_Booking_{time_of_day}" in feature_names: input_data[f"Time_of_Booking_{time_of_day}"] = 1
            if f"Customer_Loyalty_Status_{loyalty}" in feature_names: input_data[f"Customer_Loyalty_Status_{loyalty}"] = 1
            
            predicted_price = model.predict(input_data)[0]
            
            st.markdown("### Estimated Fare Results:")
            st.metric(label="Predicted Ride Cost", value=f"${predicted_price:.2f}")
            st.info(f"Estimated Cost per Minute: ${predicted_price/duration:.2f}/min")

# --- TAB 2: BUSINESS INSIGHTS ---
with tab2:
    st.subheader("Exploratory Data Analysis & Trends")
    
    selected_location = st.selectbox("Filter historical insights by Region:", ["All"] + list(df['Location_Category'].unique()))
    plot_df = df if selected_location == "All" else df[df['Location_Category'] == selected_location]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Demand & Supply Heatmap")
        heatmap_data = plot_df.pivot_table(values='Supply_and_Demand', index='Time_of_Booking', columns='Vehicle_Type', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.2f', ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.markdown("#### Ride Duration vs. Historical Cost")
        fig, ax = plt.subplots(figsize=(6, 4))
        for v_type in plot_df['Vehicle_Type'].unique():
            sub = plot_df[plot_df['Vehicle_Type'] == v_type]
            ax.scatter(sub['Expected_Ride_Duration'], sub['Historical_Cost_of_Ride'], label=v_type, alpha=0.6, edgecolor='k')
        ax.set_xlabel('Duration (mins)')
        ax.set_ylabel('Cost ($)')
        ax.legend()
        st.pyplot(fig)

# --- TAB 3: DIAGNOSTICS OF THE MODEL ---
with tab3:
    st.subheader("Random Forest Model Evaluation")
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Accuracy Score", f"{r2:.4f}")
    m2.metric("Mean Absolute Error (MAE)", f"${mae:.2f}")
    m3.metric("Root Mean Squared Error (RMSE)", f"${rmse:.2f}")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Feature Importance")
        importances = model.feature_importances_
        feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(5)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis', ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.markdown("#### Prediction Errors (Residuals)")
        diffs = np.array(y_pred) - np.array(y_test)
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['lightcoral' if d >= 0 else 'skyblue' for d in diffs]
        ax.bar(range(len(diffs)), diffs, color=colors, alpha=0.7)
        ax.axhline(0, color='black', linestyle='-')
        ax.set_ylabel('Error (Predicted - Actual) in $')
        st.pyplot(fig)