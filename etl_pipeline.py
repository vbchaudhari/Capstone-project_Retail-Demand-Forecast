#Part B — Data Architecture & ETL 

import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
BASE_PATH = r'C:\Users\Bharat\OneDrive\Desktop\Vinita Chaudhari_Capstone_RetailForecastReplenishment'
OUTPUT_PATH = os.path.join(BASE_PATH, 'data')
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Starting ETL Process...")

# --- 1. LOAD RAW FILES ---
# --- STEP 1: LOAD RAW FILES ---
stores = pd.read_csv(os.path.join(BASE_PATH, 'stores.csv'))
products = pd.read_json(os.path.join(BASE_PATH, 'products.json'))
inventory = pd.read_csv(os.path.join(BASE_PATH, 'inventory_daily.csv'))
sales = pd.read_csv(os.path.join(BASE_PATH, 'sales_daily.csv'))
pos = pd.read_csv(os.path.join(BASE_PATH, 'purchase_orders.csv'))
calendar = pd.read_csv(os.path.join(BASE_PATH, 'calendar.csv'))

# Standardize ALL dataframe columns to lowercase and remove spaces 
for df_obj in [stores, inventory, sales, pos, calendar]:
    df_obj.columns = df_obj.columns.str.strip().str.lower()


# --- 2. CLEAN & STANDARDIZE ---
# Deduplicate sales and POs 
sales = sales.drop_duplicates()
pos = pos.drop_duplicates()

# Handle missing values 
sales['units_sold'] = sales.get('units_sold', pd.Series(dtype='float')).fillna(0)
inventory['on_hand_units'] = inventory.get('on_hand_units', pd.Series(dtype='float')).fillna(0)

# Normalize category values
products['category'] = products['category'].str.lower().str.strip()

# Flag outliers (unusual spikes) [cite: 59]
sales['is_outlier'] = sales.groupby('sku_id')['units_sold'].transform(
    lambda x: (x - x.mean()).abs() > (3 * x.std())
)

# --- 3. PRODUCE CURATED DATASETS ---

# Output 1: fact_sales_store_sku_daily.csv 
#fact_sales = sales.merge(products[['sku_id', 'price', 'cost']], on='sku_id', how='left')
#fact_sales = fact_sales.merge(calendar, on='date', how='left')
#fact_sales['revenue'] = fact_sales['units_sold'] * fact_sales['price'] # [cite: 64]
#fact_sales['margin_proxy'] = fact_sales['revenue'] - (fact_sales['units_sold'] * fact_sales['cost']) # [cite: 64]
#fact_sales['day_of_week'] = pd.to_datetime(fact_sales['date']).dt.day_name() # [cite: 65]
#fact_sales.to_csv(os.path.join(OUTPUT_PATH, 'fact_sales_store_sku_daily.csv'), index=False)


# --- STEP 3: IMPROVED MERGE FOR OUTPUT 1 ---
# 1. Ensure date formats match perfectly 
sales['date'] = pd.to_datetime(sales['date'])
calendar['date'] = pd.to_datetime(calendar['date'])

# 2. Merge sales with products first [cite: 61]
fact_sales = sales.merge(products[['sku_id', 'price', 'cost']], on='sku_id', how='left')

# 3. Merge with calendar [cite: 65]
# We only take the columns we need from calendar
calendar_cols = ['date', 'promo_flag', 'holiday_flag']
# Safety check: are these columns actually in calendar?
actual_cal_cols = [c for c in calendar_cols if c in calendar.columns]
fact_sales = fact_sales.merge(calendar[actual_cal_cols], on='date', how='left')

# 4. Fill missing flags with 0 if they exist [cite: 54]
if 'promo_flag' in fact_sales.columns:
    fact_sales['promo_flag'] = fact_sales['promo_flag'].fillna(0)
if 'holiday_flag' in fact_sales.columns:
    fact_sales['holiday_flag'] = fact_sales['holiday_flag'].fillna(0)

# Calculate financial metrics [cite: 64]
fact_sales['revenue'] = fact_sales['units_sold'] * fact_sales['price']
fact_sales['margin_proxy'] = fact_sales['revenue'] - (fact_sales['units_sold'] * fact_sales['cost'])
fact_sales['day_of_week'] = fact_sales['date'].dt.day_name()

# Save the curated dataset [cite: 61]
fact_sales.to_csv(os.path.join(OUTPUT_PATH, 'fact_sales_store_sku_daily.csv'), index=False)
#print("ETL Complete: fact_sales_store_sku_daily.csv saved with flags.")




# Output 2: fact_inventory_store_sku_daily.csv 
avg_4w_demand = sales.groupby(['store_id', 'sku_id'])['units_sold'].transform(
    lambda x: x.rolling(window=28, min_periods=1).mean()
)
inventory['stockout_flag'] = (inventory['on_hand_units'] == 0).astype(int) # [cite: 69]
inventory['days_of_cover'] = inventory['on_hand_units'] / (avg_4w_demand + 0.01) # [cite: 70]
inventory.to_csv(os.path.join(OUTPUT_PATH, 'fact_inventory_store_sku_daily.csv'), index=False)

# Output 3: replenishment_inputs_store_sku.csv 
replenishment = sales.groupby(['store_id', 'sku_id']).agg(
    avg_daily_demand=('units_sold', 'mean'), # [cite: 72]
    demand_std_dev=('units_sold', 'std')    # [cite: 73]
).reset_index()

lead_times = pos.groupby('sku_id')['lead_time_days'].mean().reset_index() # [cite: 74]
replenishment = replenishment.merge(lead_times, on='sku_id', how='left').fillna(3)

# Reorder Policy Calculations 
# Safety Stock = Z (1.645) * std_dev * sqrt(lead_time) [cite: 115]
replenishment['safety_stock'] = 1.645 * replenishment['demand_std_dev'] * np.sqrt(replenishment['lead_time_days'])
# ROP = (Avg Demand * Lead Time) + Safety Stock 
replenishment['reorder_point'] = (replenishment['avg_daily_demand'] * replenishment['lead_time_days']) + replenishment['safety_stock']
replenishment['recommended_order_qty'] = replenishment['reorder_point'] * 1.1 # [cite: 117]

replenishment.to_csv(os.path.join(OUTPUT_PATH, 'replenishment_inputs_store_sku.csv'), index=False)

print(f"ETL Completed. Curated files saved in: {OUTPUT_PATH}")




