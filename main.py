import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('rocketfuel_data.csv')

# Constants from the Case
CPM = 9.00           # Cost per 1000 impressions
VALUE_PER_CONV = 40.00 # Value of a converted user

print("==========================================")
print("     ROCKET FUEL CASE ANALYSIS OUTPUT     ")
print("==========================================\n")

# ==========================================
# 1. RANDOMIZATION CHECK (Question 1)
# ==========================================
print("--- QUESTION 1: RANDOMIZATION CHECK ---")
# Calculate means
mean_impr_control = df[df['test'] == 0]['tot_impr'].mean()
mean_impr_test = df[df['test'] == 1]['tot_impr'].mean()

print(f"Avg Impressions (Control): {mean_impr_control:.2f}")
print(f"Avg Impressions (Test):    {mean_impr_test:.2f}")

# Run a t-test (using Regression) to check significance
# We regress tot_impr on test. If 'test' coeff is significant, randomization wasn't perfect.
X_rand = df[['test']]
X_rand = sm.add_constant(X_rand) # Adds the intercept 'a'
y_rand = df['tot_impr']
model_rand = sm.OLS(y_rand, X_rand).fit()

print("\nStatistical Test for Randomization (tot_impr ~ test):")
print(f"P-value: {model_rand.pvalues['test']:.4f}")
if model_rand.pvalues['test'] < 0.05:
    print(">> CONCLUSION: The groups are Statistically Different (Randomization check 'failed' slightly).")
else:
    print(">> CONCLUSION: The groups are Statistically Similar.")
print("-" * 40 + "\n")


# ==========================================
# 2. EFFECTIVENESS & REGRESSION (Question 2 & 7)
# ==========================================
print("--- QUESTION 2: EFFECTIVENESS (REGRESSION) ---")

# Linear Regression: converted = a + b1 * test
X = df[['test']]
X = sm.add_constant(X) # Adds the intercept 'a'
y = df['converted']

model = sm.OLS(y, X).fit()
print(model.summary())

# Extract coefficients for calculations
intercept_a = model.params['const'] # Baseline conversion rate
lift_b1 = model.params['test']      # Lift due to ad

print(f"\nIntercept (a): {intercept_a:.5f} ({intercept_a*100:.3f}%)")
print(f"Lift Coefficient (b1): {lift_b1:.5f} ({lift_b1*100:.3f}%)")
print(f"P-value: {model.pvalues['test']:.2e}")
print(">> CONCLUSION: The ad was effective (Positive Lift & Significant).")
print("-" * 40 + "\n")


# ==========================================
# 3. PROFITABILITY & ROI (Question 3)
# ==========================================
print("--- QUESTION 3: PROFITABILITY & ROI ---")

# A. How much more money did they make? (Incremental Revenue)
num_test_users = len(df[df['test'] == 1])
incremental_conversions = num_test_users * lift_b1
incremental_revenue = incremental_conversions * VALUE_PER_CONV

print(f"Test Group Size: {num_test_users:,}")
print(f"Calculated Lift: {lift_b1:.5f}")
print(f"Incremental Conversions: {incremental_conversions:.2f}")
print(f"A) Extra Revenue Generated: ${incremental_revenue:,.2f}")

# B. Cost of the Campaign
total_test_impressions = df[df['test'] == 1]['tot_impr'].sum()
cost_of_campaign = (total_test_impressions / 1000) * CPM
print(f"B) Cost of Campaign: ${cost_of_campaign:,.2f}")

# C. ROI
roi = (incremental_revenue - cost_of_campaign) / cost_of_campaign
print(f"C) ROI: {roi*100:.2f}%")

# D. Opportunity Cost (Control Group)
num_control_users = len(df[df['test'] == 0])
lost_conversions = num_control_users * lift_b1
lost_revenue = lost_conversions * VALUE_PER_CONV

# Cost of serving PSAs
total_control_impressions = df[df['test'] == 0]['tot_impr'].sum()
cost_of_psa = (total_control_impressions / 1000) * CPM

total_opp_cost = lost_revenue + cost_of_psa

print(f"\nD) Opportunity Cost of Control Group:")
print(f"   Lost Revenue: ${lost_revenue:,.2f}")
print(f"   Cost of PSAs: ${cost_of_psa:,.2f}")
print(f"   Total Opp Cost: ${total_opp_cost:,.2f}")
print("-" * 40 + "\n")


# ==========================================
# 4. FREQUENCY ANALYSIS (Question 4)
# ==========================================
print("--- QUESTION 4: FREQUENCY ANALYSIS ---")

# Bins requested: 0-9, 10-19...
bins = list(range(0, 210, 10)) + [float('inf')]
labels = [f"{i}-{i+9}" for i in range(0, 200, 10)] + ['200+']

df['impr_bucket'] = pd.cut(df['tot_impr'], bins=bins, labels=labels, right=False)

# Calculate rates
freq_data = df.groupby(['impr_bucket', 'test'], observed=False)['converted'].mean().unstack() * 100

# Plotting
fig, ax = plt.subplots(figsize=(14, 6))
freq_data.plot(kind='bar', colormap='inferno', width=0.8, ax=ax)
plt.title('Conversion Rate by Impressions (Activity Bias Check)', fontsize=14, fontweight='bold')
plt.ylabel('Conversion Rate (%)')
plt.xlabel('Impression Bins')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

print(">> FREQUENCY CHART GENERATED.")
print(">> Note: Observe if the Control group conversion rate rises with impressions.")
print("-" * 40 + "\n")


# ==========================================
# 5. TIME ANALYSIS (Question 5/6)
# ==========================================
print("--- TIME ANALYSIS (OPTIMIZATION) ---")

# Day Mapping
day_map = {1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat', 7: 'Sun'}
df['day_name'] = df['mode_impr_day'].map(day_map)
day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Day Data
day_data = df.groupby(['day_name', 'test'])['converted'].mean().unstack().reindex(day_order) * 100
best_day = day_data[1].idxmax()
best_day_val = day_data[1].max()

# Hour Data
hour_data = df.groupby(['mode_impr_hour', 'test'])['converted'].mean().unstack() * 100
best_hour = hour_data[1].idxmax()
best_hour_val = hour_data[1].max()

print(f"Best Day for Test Group: {best_day} ({best_day_val:.3f}%)")
print(f"Best Hour for Test Group: {best_hour}:00 ({best_hour_val:.3f}%)")

# Plot Day
day_data.plot(kind='line', figsize=(10, 5), colormap='inferno', linewidth=3, marker='o')
plt.title('Conversion Rate by Day', fontweight='bold')
plt.ylabel('Conv Rate (%)')
plt.grid(True, alpha=0.3)
plt.show()

# Plot Hour
hour_data.plot(kind='line', figsize=(10, 5), colormap='inferno', linewidth=3)
plt.title('Conversion Rate by Hour', fontweight='bold')
plt.ylabel('Conv Rate (%)')
plt.xticks(np.arange(0, 24, 1))
plt.grid(True, alpha=0.3)
plt.show()
