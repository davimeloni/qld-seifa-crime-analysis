# 📊 ***LEARNING PROJECT*** - SEIFA & Crime Rate Relationship Analysis (Queensland)

This repository contains the code and analysis for exploring the relationship between socio-economic indicators (as measured by SEIFA indexes) and violent and domestic violence (DV) crime rates in Queensland. The goal of the analysis is to understand how factors such as economic resources, education, and disadvantage impact crime rates in different regions of Queensland.
Even though there are learning explorations with the predictive aspects of models, the project is focused on the relationship and statistical significance of the factors.

---

## ⚠️ Disclaimer

This project was created as a self-guided learning exercise in data analysis and statistical modeling. While care has been taken in cleaning and analyzing the data, the results and interpretations are subject to the limitations of an individual learning process. Conclusions drawn should not be considered authoritative or used for policy-making without further validation.

---

## Project Overview

The analysis specifically focuses on the following crime categories:
- **Violent Crime**
- **Domestic Violence (DV)-Related Crime** (e.g., breaches of Domestic Violence Protection Orders)

The socio-economic factors used in this analysis are derived from the SEIFA indexes published by the Australian Bureau of Statistics. These indexes include:
- **IRSDA**: Index of Relative Socio-Economic Advantage and Disadvantage
- **IRSD**: Index of Relative Socio-Economic Disadvantage
- **IER**: Index of Education and Employment
- **IEO**: Index of Economic Opportunity

The project aims to explore how these socio-economic factors correlate with crime rates in different Queensland suburbs.

## Data Sources

- **SEIFA Indexes (2021)**  
  [Australian Bureau of Statistics](https://www.abs.gov.au/statistics/people/people-and-communities/socio-economic-indexes-areas-seifa-australia/latest-release#data-downloads)

- **Crime Data by Police Division (Jan-2018 to Mar-2025)**  
  [Queensland Government Open Data Portal](https://www.data.qld.gov.au/dataset/offence-numbers-police-divisions-monthly-from-july-2001)

## Methodology

The analysis employed several statistical methods to explore the relationships between socio-economic factors and crime rates:
1. **Pearson Correlation**: To examine the linear relationships between each SEIFA index and crime rates.
2. **Simple Linear Regression**: To quantify the strength and significance of these relationships.
3. **Multicollinearity Check**: Correlation analysis and Variance Inflation Factor (VIF) were used to identify and address multicollinearity.
4. **Lasso and Ridge Regression**: These models were employed to address multicollinearity and provide more reliable estimates.

### Performance Metrics
- **R² (R-squared)**: Used to measure how well the models explain the variance in crime rates.
- **MSE (Mean Squared Error)**: Used to evaluate the predictive accuracy of the models.

## Analysis Results

### Pearson Correlation Results
- **IER**: Moderately strong negative correlation (-0.610), indicating that areas with higher economic resources tend to have lower violent/DV crime rates.
- **IRSD**: Moderate negative correlation (-0.510), suggesting that socio-economically disadvantaged areas have higher crime rates.
- **IRSDA**: Weaker negative correlation (-0.327), showing a weaker relationship with crime rates.
- **IEO**: No significant correlation (-0.105), suggesting that education and occupation alone may not explain crime rates effectively.

### Simple Linear Regression Results
- **IER**: Explained 26% of the variance in violent/DV crime rates.
- **IRSD**: Explained 20% of the variance.
- **IRSDA**: Explained 9% of the variance.
- **IEO**: Explained less than 2%, showing a very weak relationship.

### Lasso and Ridge Regression Results

| Index   | Lasso Coefficient | Ridge Coefficient |
|---------|-------------------|-------------------|
| IRSDA   | 0.0               | 0.042             |
| IRSD    | -0.256            | -0.288            |
| IER     | -0.337            | -0.325            |
| IEO     | 0.259             | 0.237             |

### Key Insights
- Across all SEIFA indexes, the models indicate that they collectively explain approximately 30% of the variance in violent and DV crime rates. 
- The **IER** index was the most significant predictor, with higher economic resources linked to lower crime rates.
- The **IEO** index showed no significant correlation, suggesting that education and occupation alone do not explain crime rates.

## Personal Note
This report was born from that curiosity and enthusiasm. Despite being very much a beginner-level project, creating it has been both a rewarding and highly educational experience.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Australian Bureau of Statistics for providing SEIFA index data.
- Queensland Government Open Data Portal for providing crime data.



