#!/usr/bin/env python3
"""
seifa_offences_preprocessing.py

This script loads, cleans, merges, and analyzes Queensland crime data by statistical divisions, 
combining it with SEIFA socio-economic indexes. It calculates normalized crime rates per 100,000 residents, 
removes low-population areas, and visualizes the distribution of total crimes using histograms and boxplots.

Main steps:
1. Load and clean raw crime and SEIFA data
2. Group and simplify crime categories (Violent, Property, Drug, DV-related, Other)
3. Normalize crime counts by population
4. Filter out divisions with small populations (< 25th percentile)
5. Visualize the distribution of total crimes (log-scaled)
6. Save cleaned and filtered dataset to CSV

Outputs:
- Cleaned and merged dataset: `division_offences_by_seifa_indexes.csv`
- Histogram and boxplot of total crimes: saved in `../plots/`

Dependencies:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Author: Davi Santos Meloni
Created: 20-04-2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns

print("Starting SEIFA Offences Preprocessing Script...")

# load the dataset
df = pd.read_csv('../data/crimedataset.csv')
is_df = pd.read_csv('../data/indexes-scores.csv')

# Drop the Month Year column
df = df.drop(columns=["Month Year"])

# Group by the suburb (Division) and sum all crimes
crime_by_division = df.groupby("Division").sum(numeric_only=True).reset_index()

# Preview the result
print(crime_by_division['total'])
is_df.head()

# grouping crimes by violent, property, drug, domestic violence, and other categories to simplify analysis
violent_crime_cols = [
    'Homicide (Murder)', 'Other Homicide', 'Attempted Murder', 'Conspiracy to Murder',
    'Manslaughter (excl. by driving)', 'Manslaughter Unlawful Striking Causing Death',
    'Driving Causing Death', 'Assault', 'Grievous Assault', 'Serious Assault',
    'Serious Assault (Other)', "Common Assault'", 'Sexual Offences', 'Rape and Attempted Rape',
    'Other Sexual Offences', 'Robbery', 'Armed Robbery', 'Unarmed Robbery',
    'Other Offences Against the Person', 'Kidnapping & Abduction etc.', 'Extortion',
    'Life Endangering Acts', 'Voluntary Assisted Dying', 'Offences Against the Person'
]
property_crime_cols = [
    'Unlawful Entry', 'Unlawful Entry With Intent - Dwelling', 'Unlawful Entry Without Violence - Dwelling',
    'Unlawful Entry With Violence - Dwelling', 'Unlawful Entry With Intent - Shop',
    'Unlawful Entry With Intent - Other', 'Arson', 'Other Property Damage',
    'Unlawful Use of Motor Vehicle', 'Other Theft (excl. Unlawful Entry)', 'Stealing from Dwellings',
    'Shop Stealing', 'Vehicles (steal from/enter with intent)', 'Other Stealing',
    'Fraud', 'Fraud by Computer', 'Fraud by Cheque', 'Fraud by Credit Card', 'Identity Fraud',
    'Other Fraud', 'Handling Stolen Goods', 'Possess Property Suspected Stolen', 'Receiving Stolen Property',
    'Possess etc. Tainted Property', 'Other Handling Stolen Goods', 'Offences Against Property'
]
drug_offence_cols = [
    'Drug Offences', 'Trafficking Drugs', 'Possess Drugs', 'Produce Drugs',
    'Sell Supply Drugs', 'Other Drug Offences'
]
dv_related_cols = [
    'Breach Domestic Violence Protection Order'
]
other_crime_cols = [
    'Prostitution Offences', 'Found in Places Used for Purpose of Prostitution Offences',
    'Have Interest in Premises Used for Prostitution Offences', 'Stalking',
    'Knowingly Participate in Provision Prostitution Offences', 'Public Soliciting',
    'Procuring Prostitution', 'Permit Minor to be at a Place Used for Prostitution Offences',
    'Advertising Prostitution', 'Other Prostitution Offences',
    'Liquor (excl. Drunkenness)', 'Gaming Racing & Betting Offences',
    'Trespassing and Vagrancy', 'Weapons Act Offences', 'Unlawful Possess Concealable Firearm',
    'Unlawful Possess Firearm - Other', 'Bomb Possess and/or use of',
    'Possess and/or use other weapons; restricted items', 'Weapons Act Offences - Other',
    'Good Order Offences', 'Disobey Move-on Direction', 'Resist Incite Hinder Obstruct Police',
    'Fare Evasion', 'Public Nuisance', 'Stock Related Offences',
    'Traffic and Related Offences', 'Dangerous Operation of a Vehicle', 'Drink Driving',
    'Disqualified Driving', 'Interfere with Mechanism of Motor Vehicle',
    'Miscellaneous Offences', 'Other Offences'
]

# makeing a copy of the crime_by_division DataFra zame to avoid modifying the original
crime_df = crime_by_division.copy()

# Summing the relevant columns to create new columns for each crime category
crime_df["Violent Crime"] = crime_df[violent_crime_cols].sum(axis=1)
crime_df["Property Crime"] = crime_df[property_crime_cols].sum(axis=1)
crime_df["Drug Offences"] = crime_df[drug_offence_cols].sum(axis=1)
crime_df["DV-Related"] = crime_df[dv_related_cols].sum(axis=1)
crime_df["Other Crimes"] = crime_df[other_crime_cols].sum(axis=1)
crime_df.rename(columns={"total": "Total Crimes"}, inplace=True)

# Grouping the data by crime categories
# and summing the values for each category
grouped_cols = ["Violent Crime", "Property Crime", "Drug Offences", "DV-Related", "Other Crimes"]

grouped_crime_df = crime_df[[
    "Division", "Violent Crime", "Property Crime", "Drug Offences", "DV-Related", "Other Crimes", "Total Crimes"
]]

# for some reason the default total column was not being calculated correctly, so we are calculating it manually
grouped_crime_df.loc[:, "Total Crimes"] = grouped_crime_df[grouped_cols].sum(axis=1)

# merge indexes-scores.csv with the grouped crime DataFrame
merged_df = pd.merge(grouped_crime_df, is_df, how='right', left_on='Division', right_on='Localities')
merged_df.drop(columns='Localities', inplace=True)

crime_columns = ["Violent Crime", "Property Crime", "Drug Offences", "DV-Related", "Other Crimes", "Total Crimes"]

# Calculate the crime rate per 1,000 people for each crime category
# and replace the original values in the DataFrame
# to normalize the data and make it easier to compare across different regions
for col in crime_columns:
    merged_df[col] = (merged_df[col] / merged_df["Usual Resident Population"]) * 1000

# describing the population column to remove low values
merged_df['Usual Resident Population'].describe()

# after looking at the population column, decided to remove values below the 25th percentile
# to remove outliers and focus on more populated areas for more meaningful analysis
population_threshold = 482

# Number of divisions below threshold
below_threshold = merged_df[merged_df["Usual Resident Population"] < population_threshold]
print(f"Divisions below {population_threshold}: {len(below_threshold)}")

# Number of divisions above threshold
above_threshold = merged_df[merged_df["Usual Resident Population"] >= population_threshold]
print(f"Divisions above {population_threshold}: {len(above_threshold)}")

# Percentage of total
total = len(merged_df)
print(f"Percentage retained: {len(above_threshold) / total:.2%}")

print("Original dataset stats:")
print(merged_df["Usual Resident Population"].sum())

print("\nFiltered dataset stats (above threshold):")
print(above_threshold["Usual Resident Population"].sum())

# removing the divisions below the threshold
# and resetting the index for clarity
filtered_merged_df = merged_df[merged_df["Usual Resident Population"] >= population_threshold].copy()
filtered_merged_df.reset_index(drop=True, inplace=True)

# describe grouped crime data to get a summary of the data
# and check for outliers / divisions with low crime rates
# setting DV-Related and Violent Crime as the main focus for analysis
filtered_merged_df["DV-Related"].describe()
filtered_merged_df["Violent Crime"].describe()

# combining and scaling by log1p the DV-Related with Violent Crimes for better visualization of the focused analysis
filtered_merged_df['Violent_DV_Crime'] = filtered_merged_df['DV-Related'] + filtered_merged_df['Violent Crime']
filtered_merged_df['log_vdv_crime'] = np.log1p(filtered_merged_df['Violent_DV_Crime'])

# use standard scaler to scale the log of DV and Violent Crimes
scaler = StandardScaler()
filtered_merged_df['scaled_log_vdv_violence'] = scaler.fit_transform(filtered_merged_df[['log_vdv_crime']])

# check and remove outliers in the data
# Calculate the IQR for the scaled log_vdv_violence column
Q1 = filtered_merged_df['scaled_log_vdv_violence'].quantile(0.25)
Q3 = filtered_merged_df['scaled_log_vdv_violence'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_clean = filtered_merged_df[filtered_merged_df['scaled_log_vdv_violence'] <= upper_bound].copy()
print(f"Number of outliers removed: {len(filtered_merged_df) - len(df_clean)}")
print(f"New shape of the DataFrame: {df_clean.shape}")

#set Seaborn theme for cleaner visuals
sns.set_theme(style="whitegrid")

# Set up boxplot to visualise the distribution of scaled log(Violent and DV Crimes)
plt.figure(figsize=(12, 2))
sns.boxplot(
    x=df_clean['scaled_log_vdv_violence'], 
    color="lightblue",
    width=0.4,
    fliersize=4,
    linewidth=1
)

# Final touches
plt.title("Annotated Boxplot of Log-Scaled Violent and DV Crimes", fontsize=14)
plt.xlabel("log-Scaled(Violent and DV Crimes + 1)")
plt.tight_layout()
plt.savefig('../plots/violent_dv_crimes_log_scaled_boxplot.png', dpi=300, bbox_inches='tight')

# histogram of the cleaned data to see the normal distribution of the data
df_clean = df_clean.dropna()
df_clean.describe()
plt.hist(df_clean['scaled_log_vdv_violence'])
plt.title("Log-Scaled Distribution of DV and Violent Crimes per Division")
plt.xlabel("log-Scaled(DV and Violent Crimes + 1)")
plt.ylabel("Number of Divisions")
plt.savefig('../plots/violent_dv_crimes_log_scaled_distribution.png', dpi=300, bbox_inches='tight')

# save the filtered DataFrame to a new CSV file
df_clean.to_csv("../data/division_offences_by_seifa_indexes.csv", index=False)

print("Filtered dataset saved to division_offences_by_seifa_indexes.csv")
print("SEIFA Offences Preprocessing Script completed successfully.")



