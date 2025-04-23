"""
final_analysis.py

This script performs analysis on crime rates and socio-economic conditions in Queensland divisions, 
exploring the relationship between SEIFA indexes (IRSD, IER, IEO) and rates of violent and domestic violence offences. 

Main steps:
- Load the cleaned and merged dataset (division_offences_by_seifa_indexes.csv).
- Conduct exploratory data analysis and compute Pearson correlation coefficients.
- Build and evaluate simple linear regression, Lasso, and Ridge models to assess 
  the predictive power of SEIFA indexes on crime rates.
- Visualize key findings and model performance.

Outputs:
- Correlation matrices.
- Regression model summaries and evaluation metrics.
- Visualizations highlighting significant patterns and relationships.

Dependencies:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

Author: Davi Santos Meloni
Created: 23-04-2025
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

print("Starting final analysis script...")

# Load the dataset for analysis
df = pd.read_csv('../data/division_offences_by_seifa_indexes.csv')
df.describe()

# assign the SEIFA scores to a variable
seifa_scores = ['IRSDA Score', 'IRSD Score', 'IER Score', 'IEO Score']

# create variables to hold analysis results
pearson_results = {}
spearman_results = {}
lasso_results = {}
ridge_results = {}
linear_regression_results = {}

# start with the correlation matrix
# Select relevant numeric columns
cols_of_interest = ['Violent_DV_Crime', 'IRSD Score', 'IER Score', 'IEO Score', 'scaled_log_vdv_violence']
corr_matrix = df[cols_of_interest].corr()

# save the correlation matrix to a CSV file
corr_matrix.to_csv('../results/correlation_matrix.csv', index=True)

# Followed by the pearson correlation test to check for significance 
# and strength of the correlation between the SEIFA scores and the DV crime rate
# Calculate Pearson correlation coefficients and p-values for each SEIFA score against the 'Violent_DV_Crime' column
for seifa in seifa_scores:
    r, p = stats.pearsonr(df['Violent_DV_Crime'], df[seifa])
    pearson_results[seifa] = (r, p)
    print(f"{'Violent_DV_Crime'} vs {seifa}: r = {r:.3f}, p = {p:.5f} {'✅' if p < 0.05 else '❌'}")

# Save the Pearson results to a CSV file
results = {
    'IRSDA Score': pearson_results['IRSDA Score'],
    'IRSD Score': pearson_results['IRSD Score'],
    'IER Score': pearson_results['IER Score'],
    'IEO Score': pearson_results['IEO Score'],
}

corr_df = pd.DataFrame(results).T
corr_df.columns = ['r', 'p-value']
corr_df.sort_values(by='r', ascending=True, inplace=True)

# Style table
def highlight_significance(val):
    color = 'background-color: #d4edda' if val < 0.05 else ''
    return color

# Apply styling to the table
styled_df = (corr_df.style
    .map(highlight_significance, subset=['p-value'])  # Highlight p-values < 0.05
    .background_gradient(subset=['r'], cmap='coolwarm')  # Gradient for correlation values
    .format({'r': "{:.3f}", 'p-value': "{:.5f}"})  # Formatting the numbers
    .set_properties(**{'text-align': 'center'})  # Center-align text
    .set_table_styles([  # Additional styling for table borders and header
        {'selector': 'th', 'props': [('background-color', '#f7f7f7'), ('font-weight', 'bold'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('border', '1px solid #ddd'), ('padding', '8px')]},
        {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('width', '100%')]},
    ])
)

# Save the styled table to an HTML file
styled_df.to_html('../results/pearson_correlation_table.html', escape=False, index=True)

# Given the results of the Pearson correlation test, we can run a linear regression analysis for each SEIFA score against the Violent/DV crime rate.
# to further explore their relationships
X = df[seifa_scores]
y = df['scaled_log_vdv_violence']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

score_indices = {
    'IRSDA Score': 0,
    'IRSD Score': 1,
    'IER Score': 2,
    'IEO Score': 3
}

# Run the linear regression for each SEIFA score against the Violent/DV crime rate
for score in seifa_scores:
    idx = score_indices[score]
    X_single = X_scaled[:, [idx]]  # keep as 2D array
    model = LinearRegression()
    model.fit(X_single, y)
    preds = model.predict(X_single)
    
    r2 = r2_score(y, preds)
    mse = mean_squared_error(y, preds)
    linear_regression_results[score] = (r2, mse)
    print(f"{score} - R²: {r2:.3f}, MSE: {mse:.3f}")

# Plotting the linear regression results for each SEIFA score
# assign the r2 values to a variable
r2_values = {
    'IRSDA Score': linear_regression_results['IRSDA Score'][0],
    'IRSD Score': linear_regression_results['IRSD Score'][0],
    'IER Score': linear_regression_results['IER Score'][0],
    'IEO Score': linear_regression_results['IEO Score'][0]
}

# convert the r2 values to a dataframe for plotting
r2_df = pd.DataFrame(list(r2_values.items()), columns=['SEIFA Score', 'R²'])
r2_df = r2_df.sort_values(by='R²', ascending=False)

# Set style and context for a report-ready look
sns.set_theme(style="whitegrid", context="talk")

# Create a bar plot for R² values
plt.figure(figsize=(10, 6))
barplot = sns.barplot(data=r2_df, x='R²', y='SEIFA Score', palette='viridis')

# Title and axis labels
plt.title('Variance in Violent & DV Crime Rates Explained by SEIFA Indexes\n(Single Linear Regression Models)', fontsize=18, pad=20)
plt.xlabel('R² Value', fontsize=14)
plt.ylabel('SEIFA Index', fontsize=14)

# Set x-axis limits for consistency
plt.xlim(0, 0.3)

# Add value labels on bars
for index, value in enumerate(r2_df['R²']):
    plt.text(value + 0.0005, index, f"{value:.2%}", va='center', fontsize=12, color='black')

# Remove top and right spines for a cleaner look
sns.despine(left=True, bottom=True)

# Add a caption
plt.figtext(0.5, -0.005, "Note: R² represents the proportion of variance in violent and DV crime rates explained by each socio-economic index individually.", 
            ha="center", fontsize=11, color='gray')

# Tight layout for better spacing
plt.tight_layout()
#Save the plot
plt.savefig('../results/seifa_r2_single_linear_regression.png', dpi=300, bbox_inches='tight')


# Now we will check for multicollinearity between the SEIFA scores using the Variance Inflation Factor (VIF).
# To choose what models to use next
X_vif = df[['IER Score', 'IEO Score']]

vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
# Check for multicollinearity
print("Multicollinearity check:")
print(vif_data[vif_data["VIF"] > 5])  # VIF > 5 indicates multicollinearity
# I have ran this in with al possible combinations of the SEIFA scores and found that the IER and IEO scores are highly correlated with each other.

# Due to the multicollinearity between the IER and IEO scores, we will run a Ridge and Lasso regression analysis using only the IRSDA, IRSD, and IER scores as predictors.
# Now we will run a Ridge and Lasso regression analysis to see how well the SEIFA scores predict the Violent/DV crime rate.
# using LassoCV and RidgeCV to find the best alpha values for Lasso and Ridge regression

# Fit Lasso regression with cross-validation to find the best alpha
lasso_cv = LassoCV(cv=5)
lasso_cv.fit(X_scaled, y)
print("Best alpha:", lasso_cv.alpha_)
# Save Lasso coefficients
lasso_results = pd.Series(lasso_cv.coef_, index=X.columns)

# Display Lasso coefficients
print("Lasso coefficients:")
print(lasso_results)

# Fit Ridge regression with cross-validation to find the best alpha
# Note: RidgeCV does not support LassoCV's alpha selection, so we will use a range of alphas
alphas = np.logspace(-4, 1, 100)
ridge_cv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=5)
ridge_cv.fit(X_scaled, y)

print("Best alpha (Ridge):", ridge_cv.alpha_)
# Save Ridge coefficients
ridge_results = pd.Series(ridge_cv.coef_, index=X.columns)

# Display Ridge coefficients
print("Ridge coefficients:")
print(ridge_results)

# Lasso predictions
lasso_preds = lasso_cv.predict(X_scaled)
ridge_preds = ridge_cv.predict(X_scaled)

# Evaluation
print("Lasso R²:", r2_score(y, lasso_preds))
print("Ridge R²:", r2_score(y, ridge_preds))
print("Lasso MSE:", mean_squared_error(y, lasso_preds))
print("Ridge MSE:", mean_squared_error(y, ridge_preds))

# save evaluation results
lasso_preds_df = pd.DataFrame({'True': y, 'Predicted': lasso_preds})
lasso_preds_df.to_csv('../results/lasso_predictions.csv', index=False)
ridge_preds_df = pd.DataFrame({'True': y, 'Predicted': ridge_preds})
ridge_preds_df.to_csv('../results/ridge_predictions.csv', index=False)

# Plotting the coefficients of Lasso and Ridge regression
coef_df = pd.DataFrame({
    'Feature': ['IRSDA Score', 'IRSD Score', 'IER Score', 'IEO Score'],
    'Lasso': lasso_results.values,
    'Ridge': ridge_results.values
})

# Melt the DataFrame for easier plotting
coef_df_melt = coef_df.melt(id_vars='Feature', var_name='Model', value_name='Coefficient')
coef_df_melt.sort_values(by='Coefficient', ascending=True, inplace=True)

# Create the barplot
plt.figure(figsize=(10, 6))
coef_plot = sns.barplot(data=coef_df_melt, x='Coefficient', y='Feature', hue='Model', palette='viridis')

# Title and axis labels
plt.title('Comparison of Ridge and Lasso Regression Coefficients\n(Violent & DV Crime Rates)', fontsize=18, pad=20)
plt.xlabel('Coefficient Value', fontsize=14)
plt.ylabel('SEIFA Index', fontsize=14)

# Add a vertical line at zero for reference
plt.axvline(0, color='grey', linestyle='--')

# Add value labels on the bars
for container in coef_plot.containers:
    coef_plot.bar_label(container, fmt="%.2f", label_type="edge", padding=3, fontsize=11)

# Remove top and right spines for a cleaner look
sns.despine(left=True, bottom=True)

# Adjust legend
plt.legend(title='Model', loc='upper right', fontsize=12, title_fontsize=13)

# Tight layout for clean spacing
plt.tight_layout()
# Save the plot
plt.savefig('../results/ridge_vs_lasso_coefficients.png', dpi=300, bbox_inches='tight')

print("Final analysis script completed successfully.")









