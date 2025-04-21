# 📊 SEIFA & Crime Rate Analysis (Queensland)

This project investigates the relationship between socio-economic disadvantage (measured by SEIFA indexes) and crime rates—focusing specifically on domestic violence and violent crimes—across Queensland local government areas. The goal is to identify how socio-economic conditions influence serious crime, using regression analysis and data visualization.

---

## ⚠️ Disclaimer

This project was created as a learning exercise in data analysis and statistical modeling. While care has been taken in cleaning and analyzing the data, the results and interpretations are subject to the limitations of an individual learning process. Conclusions drawn should not be considered authoritative or used for policy-making without further validation.

---

## 🔍 Project Overview

**Focus**:  
- Domestic violence-related and violent crime rates

**Exclusions**:  
- Property and drug-related crimes (to isolate socio-economic impact)

**Data Sources**:  
- SEIFA 2021 Indexes from the Australian Bureau of Statistics (ABS)  
- Crime data from Queensland Government Open Data Portal

---

## 🧠 Methodology

- Grouped raw crime data into major categories  
- Normalized crime rates by population  
- Filtered out low-population areas  
- Applied log-transformation to crime rates  
- Fitted multiple linear regression using SEIFA indexes (IRSD, IER, IEO)  
- Evaluated model using R², residual diagnostics, and coefficient analysis

---

## 📈 Key Findings

- **IRSD (Disadvantage)**: Negatively correlated with DV and violent crimes  
- **IER (Education & Employment)** and **IEO (Economic Opportunity)**: Showed slight positive correlations with crime  
- All predictors were statistically significant (*p* < 0.001)  
- Regression models achieved **R² = 0.966** (DV) and **R² = 0.980** (Violent), indicating strong explanatory power  

> **Note**: While the SEIFA indexes were statistically significant predictors of crime rates, the relationships were moderate. Data points showed a general trend but were fairly spread out—suggesting that, while socio-economic disadvantage matters, other factors likely play a role too.

---

## 🛠️ How to Run

   ```bash
   git clone https://github.com/your-username/seifa-crime-analysis.git

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    pip install -r requirements.txt

    python seifa_offences_preprocessing.py

    python final_model.py
```
---

## Personal Note
This report was born from that curiosity and enthusiasm. Despite being very much a beginner-level project, creating it has been both a rewarding and highly educational experience.

## 📝 Key Findings
This project is licensed under the MIT License.



