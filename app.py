import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.ticker as ticker

st.set_page_config(page_title="Dynamic Pricing Dashboard", layout="wide", page_icon="🚗")

NUMERIC_COLUMNS = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings', 'Expected_Ride_Duration', 'Historical_Cost_of_Ride']
CATEGORICAL_COLUMNS = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Does not overwrite df
    cleaned = df.copy()

    # Standardize column names (just in case)
    cleaned.columns = cleaned.columns.str.strip()

    # Throws error if there are missing cols
    required_columns = set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)
    missing_columns = required_columns - set(cleaned.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Convert all numerical fields
    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors='coerce'
        )

    # Clean categorical values
    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = (
            cleaned[column]
            .astype('string') #convert to pandas string
            .str.strip() #strip surrounding spaces
            .str.title() #standardize capitalization
        )

    # Remove duplicate rows
    # debug code to see how many were removed
    duplicate_count = cleaned.duplicated().sum()
    print(f"Duplicate rows removed: {duplicate_count}")
    cleaned = cleaned.drop_duplicates()

    # Validate numerical ranges, checks for:
    cleaned.loc[ #if row, col val is < 0, make it nan
        cleaned['Number_of_Riders'] < 0, # negative riders makes no sense
        'Number_of_Riders'
    ] = np.nan

    cleaned.loc[
        cleaned['Number_of_Drivers'] < 0, #negative drivers
        'Number_of_Drivers'
    ] = np.nan

    cleaned.loc[
        cleaned['Expected_Ride_Duration'] <= 0, #negative duration
        'Expected_Ride_Duration'
    ] = np.nan

    cleaned.loc[
        ~cleaned['Average_Ratings'].between(1, 5), #invalid ratings
        'Average_Ratings'
    ] = np.nan

    cleaned.loc[
        cleaned['Number_of_Past_Rides'] < 0, #row
        'Number_of_Past_Rides' #col
    ] = np.nan

    cleaned.loc[
        cleaned['Historical_Cost_of_Ride'] < 0,
        'Historical_Cost_of_Ride'
    ] = np.nan

    # Mark first-time customers with a flag
    cleaned['Is_New_Customer'] = (
        cleaned['Number_of_Past_Rides'] == 0
    ).astype(int)

    # Fill only genuinely missing ratings, and point out where it was missing
    cleaned['Rating_Was_Missing'] = (
        cleaned['Average_Ratings'].isna()
    ).astype(int)

    cleaned['Average_Ratings'] = (
        cleaned['Average_Ratings']
        .fillna(cleaned['Average_Ratings'].median()) #fill with median instead, better in case of outliers
    )

    # Flag if there is no driver available
    cleaned['No_Drivers_Available'] = (
        cleaned['Number_of_Drivers'] == 0
    ).astype(int)

    essential_columns = [
        'Number_of_Riders',
        'Number_of_Drivers',
        'Number_of_Past_Rides',
        'Expected_Ride_Duration',
        'Historical_Cost_of_Ride'
    ]

    # check if there is at least 1 missing value in a row, then sum where there is
    invalid_row_count = cleaned[essential_columns].isna().any(axis=1).sum()
    print(f"Rows removed due to invalid essential values: {invalid_row_count}")

    cleaned = cleaned.dropna(subset=essential_columns)

    # Demand-to-supply ratio
    # if drivers are 0, ratio becomes nan, rather than assuming a driver
    cleaned['Supply_and_Demand'] = np.where(
        cleaned['Number_of_Drivers'] > 0, #condition
        cleaned['Number_of_Riders'] #val if true
        / cleaned['Number_of_Drivers'],
        np.nan #val if false
    )

    # No longer included in model prediction, still calcualted the same way
    cleaned['Cost_per_Minute'] = (
        cleaned['Historical_Cost_of_Ride']
        / cleaned['Expected_Ride_Duration']
    )


    # All colums returned are:
        # Is_New_Customer
        # No_Drivers_Available
        # Supply_and_Demand
        # Cost_per_Minute
        # Rating_Was_Missing
    return cleaned

def get_model_data():
    df = pd.read_csv('dynamic_pricing.csv')
    cleaned_df = clean_data(df)
    
    # Difference from old version:
    # Encodes customer loyalty status as well, so we can use it as predictor
    df_encoded = pd.get_dummies(
        cleaned_df,
        columns=[
            'Vehicle_Type',
            'Location_Category',
            'Time_of_Booking',
            'Customer_Loyalty_Status'
        ],
        drop_first=True
    )

    # Creating predictor table
    # Difference from old version:
    # remove columns directly from df_encoded
    # we drop cost_per_minute too
    X = df_encoded.drop(
        columns=[
            'Historical_Cost_of_Ride',
            'Cost_per_Minute' # also fixed typo of Cost_Per_Minute
        ]
    )

    # same logic as before, just uses column name directly
    y = df_encoded['Historical_Cost_of_Ride']

    # Difference from old version:
    # features was created before X like: 
    # features = [
    #     col for col in df_encoded.columns
    #     if col not in drop_cols
    # ]
    # 
    # in new version X is created first, and feature names are taken directly from X
    # safer, feature is guaranteed to match cols used to train model
    features = X.columns.tolist()

    # Differenc from old version:
    # Uses X instead of x, because of naming conventions, but still the same 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    random_forest_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )

    random_forest_model.fit(X_train, y_train)

    predictions = random_forest_model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    
    return cleaned_df, random_forest_model, features, X_test, y_test, predictions

# Initialization
df, model, feature_names, X_test, y_test, y_pred = get_model_data()


# --- MAIN PAGE ---
st.title("📊 Ride-Sharing Dynamic Pricing Dashboard")
st.markdown("Interactive tool for exploration and real-time algorithmic price prediction.")

tab1, tab2, tab3 = st.tabs(["💰 Price Predictor", "📈 Business Insights", "⚙️ Model Diagnostics"])

# --- TAB 1: PRICE PREDICTOR ---
with tab1:
    st.subheader("Real-Time Algorithmic Fare Estimation")
    st.markdown("#### 🚗 Ride Simulator Parameters")
    
    # Simulator inputs
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        duration = st.slider("Expected Ride Duration (minutes)", 5, 180, 25)
        riders = st.slider("Number of Riders (Demand)", 1, 100, 50)
        drivers = st.slider("Number of Available Drivers (Supply)", 0, 100, 25)
    with col_in2:
        location = st.selectbox("Location Category", ["Urban", "Suburban", "Rural"])
        time_of_day = st.selectbox("Time of Booking", ["Morning", "Afternoon", "Evening", "Night"])
        vehicle = st.selectbox("Vehicle Type", ["Economy", "Premium"])

    # Supply and demand as in predict_ride_cost
    simulated_ratio = riders / drivers if drivers > 0 else riders
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Simulated Demand/Supply Ratio", value=f"{simulated_ratio:.2f}")
            
    with col2:
        if st.button("Calculate Estimated Fare", type="primary"):

            data_dict = {col: 0 for col in feature_names}
            
            if 'Expected_Ride_Duration' in data_dict:
                data_dict['Expected_Ride_Duration'] = duration
            if 'Number_of_Riders' in data_dict:
                data_dict['Number_of_Riders'] = riders
            if 'Number_of_Drivers' in data_dict:
                data_dict['Number_of_Drivers'] = drivers
            if 'Supply_and_Demand' in data_dict:
                data_dict['Supply_and_Demand'] = simulated_ratio
            
            vehicle_col = f'Vehicle_Type_{vehicle}'
            location_col = f'Location_Category_{location}'
            time_col = f'Time_of_Booking_{time_of_day}'
            
            if vehicle_col in data_dict:
                data_dict[vehicle_col] = 1
            if location_col in data_dict:
                data_dict[location_col] = 1
            if time_col in data_dict:
                data_dict[time_col] = 1
                
            input_data = pd.DataFrame([data_dict])
            predicted_price = model.predict(input_data)[0]
            
            st.markdown("### Estimated Fare Results:")
            st.metric(label="Predicted Ride Cost", value=f"${predicted_price:.2f}")

# --- TAB 2: BUSINESS INSIGHTS ---
with tab2:
    st.subheader("Exploratory Data Analysis & Trends")
    
    # Filter for Location Category
    selected_location = st.selectbox("Filter historical insights by Region:", ["All"] + list(df['Location_Category'].unique()))
    plot_df = df if selected_location == "All" else df[df['Location_Category'] == selected_location]
    
    temp_df = plot_df.copy()
    order_times = ['Morning', 'Afternoon', 'Evening', 'Night']
    temp_df['Time_of_Booking'] = pd.Categorical(temp_df['Time_of_Booking'], categories=order_times, ordered=True)

    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("<br>", unsafe_allow_html=True)
        # Pie chart: Time of Booking Distribution
        time_of_booking_counts = plot_df['Time_of_Booking'].value_counts()
        fig1 = plt.figure(figsize=(6, 6))
        plt.pie(time_of_booking_counts, labels=time_of_booking_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'coral', 'lightgreen', 'mediumpurple' ]) 
        plt.title('Time of Booking Distribution')
        st.pyplot(fig1)

        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Bar chart: Vehicle Type Distribution
        vehicle_counts = plot_df['Vehicle_Type'].value_counts()
        fig2 = plt.figure(figsize=(5, 5))
        plt.bar(vehicle_counts.index, vehicle_counts.values, color='skyblue', edgecolor='black')
        plt.title('Vehicle Type Distribution')
        plt.xlabel('Vehicle Type')
        plt.ylabel('Number of Vehicles')
        st.pyplot(fig2)

        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Scatter plot: Expected_Ride_Duration vs. Historical_Cost_of_Ride
        fig3 = plt.figure(figsize=(6, 5))
        premium_df = plot_df[plot_df['Vehicle_Type'] == 'Premium']
        economy_df = plot_df[plot_df['Vehicle_Type'] == 'Economy']

        plt.scatter(
            economy_df['Expected_Ride_Duration'],
            economy_df['Historical_Cost_of_Ride'],
            color = 'skyblue',
            label = 'Economy',
            edgecolor = 'black',
        )
        plt.scatter(
            premium_df['Expected_Ride_Duration'],
            premium_df['Historical_Cost_of_Ride'],
            color = 'lightcoral',
            label = 'Premium',
            edgecolor = 'black',
        )
        plt.title('Pricing Efficency: Duration vs Cost by Vehicle Type')
        plt.xlabel('Expected Ride Duration in minutes')
        plt.ylabel('Cost of Ride')
        plt.legend(title='Vehicle_Type')
        plt.grid(True, linestyle='--', alpha=0.5)
        formatter = ticker.StrMethodFormatter('${x:,.2f}')
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.tight_layout()
        st.pyplot(fig3)

        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Bar chart: Average Supply and Demand by Region
        supply_demand_location = plot_df.groupby('Location_Category')['Supply_and_Demand'].mean()
        fig4 = plt.figure(figsize=(6, 5))
        plt.bar(supply_demand_location.index, supply_demand_location.values, color='skyblue', edgecolor='black')
        plt.title('Average Supply and Demand by Region')
        plt.xlabel('Region')
        plt.ylabel('Average Supply and Demand')
        st.pyplot(fig4)

    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Heatmap: Supply and Demand by Time of Booking and Vehicle Type
        heatmap_data = plot_df.pivot_table(
            values = 'Supply_and_Demand',
            index = 'Time_of_Booking',
            columns = 'Vehicle_Type',
            aggfunc = 'mean')
        fig5 = plt.figure(figsize=(6,5))
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title('Heatmap of Average Supply and Demand by Time and Vehicle')
        plt.xlabel('Vehicle Type') 
        plt.ylabel('Time of Booking')
        plt.tight_layout()
        st.pyplot(fig5)

        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Bar chart: Average Supply and Demand by Time of Booking
        average_supply_demand = temp_df.groupby('Time_of_Booking')['Supply_and_Demand'].mean()
        fig6 = plt.figure(figsize=(6, 5))
        plt.bar(average_supply_demand.index, average_supply_demand.values, color='skyblue', edgecolor='black')
        plt.ylim(0, 4)
        plt.title('Average Supply and Demand by Time of Booking')
        plt.xlabel('Time of Booking')
        plt.ylabel('Average Supply and Demand') 
        st.pyplot(fig6)

        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Bar chart: Average Cost of Ride by Time of Booking
        avg_revenue = temp_df.groupby('Time_of_Booking')['Historical_Cost_of_Ride'].mean()
        fig7 = plt.figure(figsize=(6, 5))
        ax = avg_revenue.plot(kind='bar', color='skyblue', edgecolor='black', ax=plt.gca())
        formatter = ticker.StrMethodFormatter('${x:,.2f}')
        ax.yaxis.set_major_formatter(formatter)
        plt.title('Average Ride Cost by Time of Booking') 
        plt.xlabel('Time of Booking')
        plt.ylabel('Average Cost')
        plt.xticks(rotation=0) 
        st.pyplot(fig7)

        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Line plot: Average Cost per Minute by Location_Category and Vehicle_Type
        summary_df = plot_df.groupby(['Location_Category', 'Vehicle_Type'])['Cost_per_Minute'].mean().unstack()
        fig8 = plt.figure(figsize=(8, 6))
        for vehicle_type in summary_df.columns:
            plt.plot(summary_df.index, summary_df[vehicle_type], marker='o', label=vehicle_type)
        plt.title('Average Cost per Minute by Region and Vehicle Type')
        plt.xlabel('Region')
        plt.ylabel('Average Cost per Minute')
        plt.legend(title='Vehicle Type')
        plt.ylim(2, 5)
        plt.grid()
        st.pyplot(fig8)

# --- TAB 3: MODEL DIAGNOSTICS ---
with tab3:
    st.subheader("Random Forest Model Evaluation")
    
    r2 = round(r2_score(y_test, y_pred), 2)
    mae = round(mean_absolute_error(y_test, y_pred), 0)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 0)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Accuracy Score", f"{r2:.2f}")
    m2.metric("Mean Absolute Error (MAE)", f"${mae:.0f}")
    m3.metric("Root Mean Squared Error (RMSE)", f"${rmse:.0f}")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top 10 Feature Importances")
        # Feature Importance bar plot
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        top_features = feature_importance.head(10)
        
        fig9 = plt.figure(figsize=(8, 6))
        plt.barh(top_features['Feature'], top_features['Importance'])
        plt.gca().invert_yaxis()
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        st.pyplot(fig9)
        
    with col2:
        st.markdown("#### Predicted vs Actual Ride Prices")
        # Predicted vs Actual scatter plot
        fig10 = plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)

        min_price = min(y_test.min(), y_pred.min())
        max_price = max(y_test.max(), y_pred.max())

        plt.plot([min_price, max_price], [min_price, max_price], linestyle='--')
        plt.title('Predicted vs Actual Ride Prices')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.grid(True)
        st.pyplot(fig10)
