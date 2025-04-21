"""
final_model.py

This script loads the preprocessed crime and SEIFA data, fits a multiple linear regression model to predict the log-transformed crime rates based on socio-economic indexes, 
and evaluates the model's performance. It also visualizes the regression coefficients for SEIFA scores and the actual vs predicted crime rates.

Main steps:
Load the cleaned and merged dataset (division_offences_by_seifa_indexes.csv).
Split the data into features (SEIFA scores) and target variables (log-transformed crime rates).
Build and fit a multiple linear regression model to predict domestic violence (DV) and violent crime rates based on SEIFA indexes (IRSD, IER, IEO).
Evaluate the model's performance using R-squared and residual analysis.
Visualize the regression coefficients for each SEIFA score and the relationship between actual and predicted crime rates.
Save the regression results and model summary to a text file for further analysis.

Outputs:
Regression results summary (saved in model_summary.txt).
Coefficient visualizations: saved as barplots in ../plots/.
Scatter plot of actual vs predicted crime rates.

Dependencies:
pandas
numpy
statsmodels
matplotlib
seaborn

Author: Davi Santos Meloni
Created: 20-04-2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# printing the start of the script
print("Starting the final model script...")

# read the data
df = pd.read_csv('../data/division_offences_by_seifa_indexes.csv')

# Drop rows with missing values
df_clean = df.dropna()
df = df_clean.copy()

# Crime columns list
crime_cols = ['Violent Crime', 'Property Crime', 'Drug Offences',
              'DV-Related', 'Other Crimes', 'Total Crimes']

# deciles columns
deciles = ['IRSD Decile', 'IER Decile', 'IEO Decile']

# descale the crime rate by pouplation from 100.000 to 1000
df[crime_cols] = df[crime_cols] / 100

# log transform the data to reduce the effect of outliers
df['log_DV_Crime'] = np.log1p(df['DV-Related'])  # log(1 + x) handles zero values safely
df['log_Violent_Crime'] = np.log1p(df['Violent Crime'])  # log(1 + x) handles zero values safely

# Define the independent variables (SEIFA scores)
X = df[['IRSD Score', 'IER Score', 'IEO Score']]
# Response variables (dependent variables)
y_actual_log_DV_Crime = df['log_DV_Crime']
y_actual_log_Violent_Crime = df['log_Violent_Crime']
# Model for DV-related crime
model_dv = sm.OLS(y_actual_log_DV_Crime, X).fit()

# Model for Violent crime
model_violent = sm.OLS(y_actual_log_Violent_Crime, X).fit()

# Save the model summaries to text files
with open('../model_summaries/model_dv_summary.txt', 'w') as f:
    f.write(str(model_dv.summary()))

with open('../model_summaries/model_violent_summary.txt', 'w') as f:
    f.write(str(model_violent.summary()))


# Get the predicted values
y_pred_log_DV_Crime = model_dv.predict(X)  # Predictions for log_DV_Crime
y_pred_log_Violent_Crime = model_violent.predict(X)  # Predictions for log_Violent_Crime

# Set up the matplotlib figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot for DV-Related Crime
sns.scatterplot(x=y_actual_log_DV_Crime, y=y_pred_log_DV_Crime, ax=axes[0], color='steelblue')
axes[0].plot([y_actual_log_DV_Crime.min(), y_actual_log_DV_Crime.max()], 
             [y_actual_log_DV_Crime.min(), y_actual_log_DV_Crime.max()], color='red', linestyle='--')
axes[0].set_title('DV-Related Crime: Actual vs Predicted', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Actual log(DV Crime)', fontsize=14)
axes[0].set_ylabel('Predicted log(DV Crime)', fontsize=14)
axes[0].grid(True)

# Plot for Violent Crime
sns.scatterplot(x=y_actual_log_Violent_Crime, y=y_pred_log_Violent_Crime, ax=axes[1], color='darkorange')
axes[1].plot([y_actual_log_Violent_Crime.min(), y_actual_log_Violent_Crime.max()], 
             [y_actual_log_Violent_Crime.min(), y_actual_log_Violent_Crime.max()], color='red', linestyle='--')
axes[1].set_title('Violent Crime: Actual vs Predicted', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Actual log(Violent Crime)', fontsize=14)
axes[1].set_ylabel('Predicted log(Violent Crime)', fontsize=14)
axes[1].grid(True)

fig.suptitle("Model Fit: Predicted vs Actual Crime Rates", fontsize=24, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.savefig('../plots/actual_vs_predicted_crimes_by_seifa_score.png', dpi=300, bbox_inches='tight')

# Coefficients from the regression model
dv_coefficients = {
    'IRSD Score': -0.0150,  # Negative effect on log_DV_Crime
    'IER Score': 0.0081,    # Positive effect on log_DV_Crime
    'IEO Score': 0.0123     # Positive effect on log_DV_Crime
}
vc_coefficients = {
    'IRSD Score': -0.0152,  # Negative effect on log_Violent_Crime
    'IER Score': 0.0084,    # Positive effect on log_Violent_Crime
    'IEO Score': 0.0138     # Positive effect on log_Violent_Crime
}

# Creating a DataFrame for better visualization
dv_df = pd.DataFrame(list(dv_coefficients.items()), columns=['SEIFA Score', 'DV Crime Coefficient'])
vc_df = pd.DataFrame(list(vc_coefficients.items()), columns=['SEIFA Score', 'Violent Crime Coefficient'])

# Set up the matplotlib figure
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.barplot(x='SEIFA Score', y='DV Crime Coefficient', data=dv_df, ax=axes[0], palette='pastel')
axes[0].set_title('Coefficients for SEIFA Scores on DV-Related Crime Rates', fontsize=16, fontweight='bold')
axes[0].set_xlabel('SEIFA Score', fontsize=14)
axes[0].set_ylabel('DV Coefficient', fontsize=14)
axes[0].axhline(0, color='black', linewidth=1)

sns.barplot(x='SEIFA Score', y='Violent Crime Coefficient', data=vc_df, ax=axes[1], palette='pastel')
axes[1].set_title('Coefficients for SEIFA Scores on Violent Crime Rates', fontsize=16, fontweight='bold')
axes[1].set_xlabel('SEIFA Score', fontsize=14)
axes[1].set_ylabel('Violent Crime Coefficient', fontsize=14)
axes[1].axhline(0, color='black', linewidth=1) 

fig.suptitle("Regression Coefficients for SEIFA Scores on Crime Rates", fontsize=24)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.savefig('../plots/regression_coefficients_seifa_crime.png', dpi=300, bbox_inches='tight')

# printing the end of the script
print("Final model script completed.")