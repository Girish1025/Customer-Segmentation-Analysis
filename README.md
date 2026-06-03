# Customer Campaign Response Prediction

## Project Overview

This project analyzes customer demographics, purchase behavior, and prior campaign engagement to predict whether a customer will respond to a marketing campaign. The workflow includes exploratory data analysis, feature engineering, imbalance handling, machine learning model development, hyperparameter tuning, cross-validation, and SHAP-based model interpretation.

The target variable is `response`, where:

- `1` = Customer responded to the campaign
- `0` = Customer did not respond to the campaign

## Objective

The goal of this project is to help improve marketing campaign targeting by identifying customer segments that are more likely to respond to offers.

Key questions explored include:

- Which customer groups are most likely to respond to campaigns?
- How do income, recency, age group, marital status, and education affect campaign response?
- Does previous campaign acceptance increase the likelihood of responding again?
- Which machine learning model performs best for predicting customer response?
- Which features have the strongest influence on predictions?

## Project Structure

```text
Customer-Campaign-Response-Prediction/
│
├── data/
│   └── marketing_campaign.xlsx
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── hyperparameter_tuning.py
│   └── shap_analysis.py
│
├── outputs/
│   ├── figures/
│   └── models/
│
├── main.py
├── requirements.txt
├── README.md
├── ABOUT_GITHUB.txt
└── .gitignore
```
## Author

**Girish S Chandrappa**
