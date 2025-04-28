# Author: Ahmad Hamdaan
# Project: World Energy Consumption Predictive Model (Per-Country and Global Totals)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ------------------------------
# 1. Load and Prepare Dataset
# ------------------------------
df = pd.read_csv("World Energy Consumption.csv")
df.columns = df.columns.str.lower()
df_clean = df.dropna()

# Ensure correct dtypes
df_clean.loc[:, 'year'] = df_clean['year'].astype(int)

# ------------------------------
# 2. Unique Country List
# ------------------------------
countries = df_clean['country'].unique()

# Store all predictions
all_predictions = []

# ------------------------------
# 3. Predict per Country
# ------------------------------
for country in countries:
    country_df = df_clean[df_clean['country'] == country]
    
    # Use data up to 2015
    train_df = country_df[country_df['year'] <= 2015]
    if len(train_df) < 2:
        continue  # Not enough data
    
    X_train = train_df[['year']]
    y_train = train_df['electricity_demand']
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict 2016–2025
    future_years = pd.DataFrame({'year': list(range(2016, 2026))})
    predictions = model.predict(future_years)
    
    # Store predictions
    country_pred = future_years.copy()
    country_pred['country'] = country
    country_pred['predicted_demand'] = predictions
    all_predictions.append(country_pred)

# Combine all country-level predictions
predicted_df = pd.concat(all_predictions)

# ------------------------------
# 4. Aggregate Total Demand Per Year
# ------------------------------
# Historical totals (up to 2015)
historical_totals = (
    df_clean[df_clean['year'] <= 2015]
    .groupby('year')['electricity_demand']
    .sum()
    .reset_index()
    .rename(columns={'electricity_demand': 'total_demand'})
)

# Predicted totals (2016–2025)
predicted_totals = (
    predicted_df
    .groupby('year')['predicted_demand']
    .sum()
    .reset_index()
    .rename(columns={'predicted_demand': 'total_demand'})
)

# Combine full demand history + forecast
full_demand = pd.concat([historical_totals, predicted_totals], ignore_index=True)

# ------------------------------
# 5. Evaluate Global Model Performance
# ------------------------------
# Create a global model using sum of each year as one data point
X_global = historical_totals[['year']]
y_global = historical_totals['total_demand']
global_model = LinearRegression()
global_model.fit(X_global, y_global)

# Predict for 2016–2025 using global model
X_future_global = predicted_totals[['year']]
y_pred_global = global_model.predict(X_global)

# Evaluate performance on historical (train) global totals
r2 = r2_score(y_global, y_pred_global)
rmse = mean_squared_error(y_global, y_pred_global) ** 0.5

print("\n[Global Model Evaluation] (on historical totals up to 2015):")
print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} (TWh)")

# ------------------------------
# 6. Plot Total Electricity Demand
# ------------------------------
plt.figure(figsize=(10, 6))
plt.plot(historical_totals['year'], historical_totals['total_demand'], label='Historical (Actual)', color='blue')
plt.plot(predicted_totals['year'], predicted_totals['total_demand'], label='Forecast (2016–2025)', linestyle='--', color='red')
plt.title("Global Total Electricity Demand Forecast (2016–2025)")
plt.xlabel("Year")
plt.ylabel("Total Electricity Demand (TWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()  # 

# ------------------------------
# 7. Show Output Tables
# ------------------------------
print("\n Full Global Electricity Demand (Actual + Predicted):")
print(full_demand)

# ------------------------------
# 8. Export Results (Optional)
# ------------------------------
# predicted_df.to_csv("country_predictions_2016_2025.csv", index=False)
# full_demand.to_csv("total_demand_by_year.csv", index=False)

# ------------------------------
# 9. Additional Insights from Forecast
# ------------------------------

start_year = 2015
end_year = 2025

# Get start and end values
start_demand = historical_totals[historical_totals['year'] == start_year]['total_demand'].values[0]
end_demand = predicted_totals[predicted_totals['year'] == end_year]['total_demand'].values[0]

# Calculate Compound Annual Growth Rate (CAGR)
years = end_year - start_year
cagr = ((end_demand / start_demand) ** (1 / years)) - 1

# Peak and minimum forecast years
peak_year = predicted_totals.loc[predicted_totals['total_demand'].idxmax()]
min_year = predicted_totals.loc[predicted_totals['total_demand'].idxmin()]

# Total forecasted increase
total_change = end_demand - start_demand

print("\n Forecast Summary (2016–2025):")
print(f"- Starting Demand in {start_year}: {start_demand:,.2f} TWh")
print(f"- Forecasted Demand in {end_year}: {end_demand:,.2f} TWh")
print(f"- Total Increase (2015–2025): {total_change:,.2f} TWh")
print(f"- Compound Annual Growth Rate (CAGR): {cagr * 100:.2f}% per year")
print(f"- Peak Forecast Year: {int(peak_year['year'])} ({peak_year['total_demand']:,.2f} TWh)")
print(f"- Lowest Forecast Year: {int(min_year['year'])} ({min_year['total_demand']:,.2f} TWh)")

# ------------------------------
# 10. Comparison with Historical Data
# ------------------------------

# Last 10 years of historical demand (2006–2015)
historical_last_decade = historical_totals[historical_totals['year'].between(2006, 2015)]
hist_avg = historical_last_decade['total_demand'].mean()

# Forecast decade (2016–2025)
forecast_avg = predicted_totals['total_demand'].mean()

# Entire historical average
full_hist_avg = historical_totals['total_demand'].mean()

# Change in average demand
avg_change = forecast_avg - hist_avg
percent_change = (avg_change / hist_avg) * 100

print("\n Comparison with Historical Data:")
print(f"- Avg. Demand (2006–2015): {hist_avg:,.2f} TWh")
print(f"- Avg. Forecasted Demand (2016–2025): {forecast_avg:,.2f} TWh")
print(f"- Avg. Change: {avg_change:,.2f} TWh ({percent_change:.2f}%)")
print(f"- Overall Historical Avg. (before 2016): {full_hist_avg:,.2f} TWh")

# Trend Analysis
if percent_change > 5:
    trend = "an accelerating trend in global energy demand."
elif percent_change < -5:
    trend = "a deceleration in global energy demand."
else:
    trend = "a relatively steady trend in global energy demand."

print(f"\nTrend: {trend}")

# ------------------------------
# 11. Compare Predictions to Actuals (2016–2022)
# ------------------------------
# Assuming `comparison_df` contains actual vs predicted data for 2016–2022
plt.figure(figsize=(10, 6))
plt.plot(comparison_df['year'], comparison_df['actual_demand'], label='Actual', color='blue', marker='o')
plt.plot(comparison_df['year'], comparison_df['predicted_demand'], label='Predicted', color='orange', linestyle='--', marker='o')
plt.fill_between(comparison_df['year'], comparison_df['actual_demand'], comparison_df['predicted_demand'], alpha=0.2, color='gray')
plt.title("Actual vs. Predicted Global Electricity Demand (2016–2022)")
plt.xlabel("Year")
plt.ylabel("Electricity Demand (TWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()  # 

# ------------------------------
# 12. Summary Plot with Annotations
# ------------------------------
plt.figure(figsize=(12, 7))

# Plot historical (actual) demand
plt.plot(historical_totals['year'], historical_totals['total_demand'], label='Historical Demand', color='blue', marker='o')

# Plot forecasted demand
plt.plot(predicted_totals['year'], predicted_totals['total_demand'], label='Predicted Demand', color='red', linestyle='--', marker='o')

# Highlight peak and min forecast years
plt.scatter(peak_year['year'], peak_year['total_demand'], color='green', s=100, zorder=5, label='Peak Forecast Year')
plt.scatter(min_year['year'], min_year['total_demand'], color='purple', s=100, zorder=5, label='Lowest Forecast Year')

# Annotate CAGR
plt.text(end_year - 6, end_demand + 4000,
         f"CAGR (2015–{end_year}): {cagr*100:.2f}%",
         fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))

# Annotate peak
plt.text(peak_year['year'], peak_year['total_demand'] + 2000,
         f"Peak: {peak_year['year']} ({peak_year['total_demand']:.0f} TWh)",
         fontsize=9, ha='center', bbox=dict(boxstyle="round", fc="white"))

# Annotate min
plt.text(min_year['year'], min_year['total_demand'] - 3000,
         f"Min: {min_year['year']} ({min_year['total_demand']:.0f} TWh)",
         fontsize=9, ha='center', bbox=dict(boxstyle="round", fc="white"))

# Labels and formatting
plt.title("Global Electricity Demand Forecast (Actual + Predicted + Key Insights)", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Total Electricity Demand (TWh)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()  # 
