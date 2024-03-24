#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import base64

days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Load and preprocess the data
# @st.cache
def load_data():
    walmart_data = pd.read_csv("RetailAnalyticsWalmart.csv")
    
    # Cleaning the 'Prescription Value' column
    walmart_data['Prescription Value'] = walmart_data['Prescription Value'].replace('#VALUE!', float('nan'))
    walmart_data.dropna(subset=['Prescription Value'], inplace=True)
    walmart_data['Prescription Value'] = walmart_data['Prescription Value'].astype(float)
    
    # Extracting temporal features from 'Date Dispensed'
#     walmart_data['Date Dispensed'] = pd.to_datetime(walmart_data['Date Dispensed'], format='mixed')
#     walmart_data['Date Dispensed'] = pd.to_datetime(walmart_data['Date Dispensed'], format='%m/%d/%Y %H:%M')
    walmart_data['Date Dispensed'] = pd.to_datetime(walmart_data['Date Dispensed'], format='%d/%m/%Y %H:%M')
    #     walmart_data['Date Dispensed'] = pd.to_datetime(walmart_data['Date Dispensed'])
    walmart_data['Hour'] = walmart_data['Date Dispensed'].dt.hour
    walmart_data['Day'] = walmart_data['Date Dispensed'].dt.day_name()
    walmart_data['Week'] = walmart_data['Date Dispensed'].dt.isocalendar().week

    return walmart_data

data = load_data()

# 2. Additional Transformations
def additional_transformations(data):
    
    # Convert 'Date Dispensed' to datetime format
    data['Date Dispensed'] = pd.to_datetime(data['Date Dispensed'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

    # Filter out rows with null 'Date Dispensed' values
    data = data.dropna(subset=['Date Dispensed'])
    
    avg_monthly_sales = data.groupby(['Branch', 'Drug Name']).agg({
        'Quantity Dispensed': 'mean',
        'Prescription Value': 'mean'
    }).reset_index()

    avg_monthly_sales['Stocks on Hand'] = avg_monthly_sales['Quantity Dispensed'] * 3 * np.random.uniform(0.8, 1.2, len(avg_monthly_sales))

    data = data.merge(avg_monthly_sales[['Branch', 'Drug Name', 'Stocks on Hand']], on=['Branch', 'Drug Name'], how='left')
    
    return data

data = additional_transformations(data)

# Streamlit app with adjusted title size using inline CSS
st.markdown("<h1 style='text-align: center; font-size: 20px;'>Walmart Pharmacies Customer Traffic Patterns Analysis</h1>", unsafe_allow_html=True)

logo_base64 = get_image_base64("walmart.png")
st.markdown(
    f"<div style='text-align: center'><img src='data:image/png;base64,{logo_base64}'></div>",
    unsafe_allow_html=True,
)


# User input for filters
selected_branch = st.selectbox("Select a branch for analysis:", ["All"] + sorted(data['Branch'].unique()))
from_date = st.date_input("From Date", min_value=data['Date Dispensed'].min().date(), max_value=data['Date Dispensed'].max().date(), value=data['Date Dispensed'].min().date())
to_date = st.date_input("To Date", min_value=from_date, max_value=data['Date Dispensed'].max().date(), value=data['Date Dispensed'].max().date())
selected_drug = st.selectbox("Select a drug for analysis:", ["All"] + sorted(data['Drug Name'].unique()))

# Filter data based on user input
filtered_data = data.copy()
if selected_branch != "All":
    filtered_data = filtered_data[filtered_data['Branch'] == selected_branch]
if selected_drug != "All":
    filtered_data = filtered_data[filtered_data['Drug Name'] == selected_drug]
filtered_data = filtered_data[(filtered_data['Date Dispensed'].dt.date >= from_date) & (filtered_data['Date Dispensed'].dt.date <= to_date)]

# Branch-wise Sales Performance on filtered data
branch_performance = filtered_data.groupby('Branch').agg({
    'Prescription Value': ['sum', 'mean'],
    'Quantity Dispensed': ['sum', 'mean']
}).reset_index()
branch_performance[('Prescription Value', 'sum')] = branch_performance[('Prescription Value', 'sum')].apply(lambda x: "${:,.2f}".format(x))
branch_performance[('Prescription Value', 'mean')] = branch_performance[('Prescription Value', 'mean')].apply(lambda x: "${:,.2f}".format(x))
branch_performance[('Quantity Dispensed', 'sum')] = branch_performance[('Quantity Dispensed', 'sum')].apply(lambda x: "{:,.0f}".format(x))
branch_performance[('Quantity Dispensed', 'mean')] = branch_performance[('Quantity Dispensed', 'mean')].apply(lambda x: "{:,.0f}".format(x))

branch_performance.columns = ['_'.join(col).strip() for col in branch_performance.columns.values]
branch_performance = branch_performance.rename(columns={'Branch_': 'Branch'})

# Function to create heatmap
def generate_heatmap(data, column):
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = data.groupby(['Hour', 'Day']).agg({column:'sum'}).unstack().fillna(0)
    heatmap_data.columns = heatmap_data.columns.get_level_values(1)
    
    # Ensure all days are present in heatmap data
    for day in days_order:
        if day not in heatmap_data.columns:
            heatmap_data[day] = 0
    heatmap_data = heatmap_data[days_order]
    
     # # Calculate daily totals
    daily_totals = heatmap_data.sum()
    heatmap_data = heatmap_data.append(pd.DataFrame([daily_totals], index=['Total']))
        
    heatmap_data = heatmap_data.sort_index(ascending=False)  # Order hours from morning to evening
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use appropriate formatting for metric
    if metric == 'Prescription Value':
        sns.heatmap(heatmap_data, cmap="YlGnBu", linewidths=0.5, ax=ax, annot=True, cbar=False, fmt="${:,.0f}")
    else:  # 'Quantity Dispensed'
        sns.heatmap(heatmap_data, cmap="YlGnBu", linewidths=0.5, ax=ax, annot=True, cbar=False, fmt="{:,.0f}")
        
    plt.yticks(rotation=0)
    plt.title(f"Heatmap of {metric} by Hour and Day")
    return fig

#     fig, ax = plt.subplots(figsize=(12, 8))
#     if column == 'Prescription Value':
#         sns.heatmap(heatmap_data, cmap="YlGnBu", linewidths=0.5, ax=ax, annot=True, cbar=False, fmt="g", annot_kws={"size": 10, "weight": "bold"})
#         # Format the annotations as currency
#         for t in ax.texts:
#             t.set_text('${:,.2f}'.format(float(t.get_text())))
#     else:
#         sns.heatmap(heatmap_data, cmap="YlGnBu", linewidths=0.5, ax=ax, annot=True, cbar=False, fmt="g", annot_kws={"size": 10, "weight": "bold"})
#         # Format the annotations with comma
#         for t in ax.texts:
#             t.set_text('{:,.0f}'.format(float(t.get_text())))
#     ax.set_ylabel('Hour')
#     ax.set_xlabel('Day')
#     return fig

# Dynamic title based on selected values with reduced font size
title_branch = f"Branch: {selected_branch}" if selected_branch != "All" else "All Branches"
title_drug = f", Drug: {selected_drug}" if selected_drug != "All" else ""
title_date_range = f", Date Range: {from_date} to {to_date}"
st.markdown(f"<h3 style='font-size: 16px;'>Analysis for {title_branch}{title_drug}{title_date_range}</h3>", unsafe_allow_html=True)

# Heatmap for Quantity Dispensed
st.subheader("Heatmap for Quantity Dispensed")
fig1 = generate_heatmap(filtered_data, 'Quantity Dispensed')
st.pyplot(fig1)

# Heatmap for Prescription Value
st.subheader("Heatmap for Prescription Value")
fig2 = generate_heatmap(filtered_data, 'Prescription Value')
st.pyplot(fig2)

# Dynamic Insights
st.header("Key Insights")
peak_hour_value = filtered_data.groupby('Hour')['Prescription Value'].sum().idxmax()
peak_hour_quantity = filtered_data.groupby('Hour')['Quantity Dispensed'].sum().idxmax()
peak_day_value = filtered_data.groupby('Day')['Prescription Value'].sum().idxmax()
peak_day_quantity = filtered_data.groupby('Day')['Quantity Dispensed'].sum().idxmax()
popular_drugs = filtered_data.groupby('Drug Name').size().sort_values(ascending=False).head(5)

st.write(f"Peak hour for sales by Prescription Value: **{int(peak_hour_value)}:00**")
st.write(f"Peak hour for sales by Quantity Dispensed: **{int(peak_hour_quantity)}:00**")
st.write(f"Peak day for sales by Prescription Value: **{peak_day_value}**")
st.write(f"Peak day for sales by Quantity Dispensed: **{peak_day_quantity}**")
st.write("Most commonly purchased drugs:")
for i, (drug, count) in enumerate(popular_drugs.items(), 1):
    st.write(f"{i}. **{drug}** with **{count}** purchases")

# Dynamic Hourly Distribution Visualization
st.subheader("Hourly Distribution of Sales")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
filtered_data.groupby('Hour')['Quantity Dispensed'].sum().plot(kind='bar', ax=ax1, title="Quantity Dispensed")
filtered_data.groupby('Hour')['Prescription Value'].sum().plot(kind='bar', ax=ax2, title="Prescription Value ($)")
plt.tight_layout()
st.pyplot(fig)

# Dynamic Daily Distribution Visualization
st.subheader("Daily Distribution of Sales")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
filtered_data.groupby('Day')['Quantity Dispensed'].sum().reindex(days_order).plot(kind='bar', ax=ax1, title="Quantity Dispensed")
filtered_data.groupby('Day')['Prescription Value'].sum().reindex(days_order).plot(kind='bar', ax=ax2, title="Prescription Value ($)")
plt.tight_layout()
st.pyplot(fig)

# Recommendations and Insights
st.header("Recommendations & Insights")

# Suggested Opening Days and Hours
def suggested_hours_days(df):
    # Suggested Days
    day_sales = df.groupby('Day')['Prescription Value'].sum().reindex(days_order)
    suggested_days = day_sales[day_sales > day_sales.median()].index.tolist()
    
    # Suggested Hours
    hour_sales = df.groupby('Hour')['Prescription Value'].sum()
    suggested_hours = hour_sales[hour_sales > hour_sales.median()].index.tolist()
    
    return suggested_days, suggested_hours

suggested_days, suggested_hours = suggested_hours_days(filtered_data)
st.write(f"Suggested Opening Days for optimal sales: **{', '.join(suggested_days)}**")
st.write(f"Suggested Opening Hours for optimal sales: **{', '.join([str(int(hour))+':00' for hour in suggested_hours])}**")

# Staffing Recommendations
def staffing_recommendations(df):
    hour_sales = df.groupby('Hour')['Prescription Value'].sum()
    
    # Lower traffic hours (below 25th percentile)
    low_traffic_hours = hour_sales[hour_sales < hour_sales.quantile(0.25)].index.tolist()
    
    # Higher traffic hours (above 75th percentile)
    high_traffic_hours = hour_sales[hour_sales > hour_sales.quantile(0.75)].index.tolist()
    
    return low_traffic_hours, high_traffic_hours

low_traffic_hours, high_traffic_hours = staffing_recommendations(filtered_data)
st.write(f"Consider reducing staff during these hours: **{', '.join([str(int(hour))+':00' for hour in low_traffic_hours])}**")
st.write(f"Consider increasing staff during these hours: **{', '.join([str(int(hour))+':00' for hour in high_traffic_hours])}**")

# Streamlit application starts here
st.title("Retail Pharmacy Analysis")

# Display the branch-wise sales performance
st.write("## Branch-wise Sales Performance")
st.table(branch_performance)

# Display insights into the slow-moving stock for the selected branch
st.write("## Slow Moving Stock Analysis")
branch_selected = st.selectbox("Select a branch for detailed analysis:", branch_performance['Branch'].unique())
slow_moving_data = data[data['Branch'] == branch_selected]
slow_moving_threshold = 0.1
slow_moving_data['Sales-to-Stock Ratio'] = slow_moving_data['Quantity Dispensed'] / slow_moving_data['Stocks on Hand']
slow_moving_candidates = slow_moving_data.groupby('Drug Name').agg({
    'Sales-to-Stock Ratio': 'mean',
    'Stocks on Hand': 'mean'
}).sort_values(by='Sales-to-Stock Ratio').reset_index()
slow_moving_candidates_filtered = slow_moving_candidates[slow_moving_candidates['Sales-to-Stock Ratio'] <= slow_moving_threshold]

if not slow_moving_candidates_filtered.empty:
    st.write("Here are the slow-moving products for the selected branch:")
    st.table(slow_moving_candidates_filtered)
else:
    st.write(f"The {branch_selected} branch does not have any slow-moving products based on the defined criteria.")
    
# Dynamic recommendations based on data insights
st.write("## Recommendations")

if not slow_moving_candidates_filtered.empty:
    st.write(f"The {branch_selected} branch has slow-moving products. Consider the following actions:")
    st.write("1. Run promotional activities to boost sales for those products.")
    st.write("2. Monitor inventory levels and consider redistributing stock to other branches if necessary.")
    st.write("3. Optimize stock levels based on demand and explore opportunities for cross-selling.")
else:
    st.write(f"The {branch_selected} branch does not have any slow-moving products based on the defined criteria. Keep up the good inventory management!")




