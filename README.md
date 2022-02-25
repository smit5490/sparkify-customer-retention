# Customer Retention Modeling using Spark, SHAP, and Streamlit  

## Project Definition
This project is based on a fictitious digital music company, Sparkify. Customers use their platform to stream music 
using either a free or paid subscription. Their data engineering team has set-up user monitoring logs that tracks
customer behavior. Sparkify would like to explore the use of these logs to determine if they can predict which customers
are likely to cancel their service, and why. With this information, the business can create a customer retention 
strategy to increase customer loyalty.

## Project Deliverable
The goal of this project is to build and select a model with the highest f1-score. Using this model, create an application
to identify high risk customers and provide insights into what contributes to risk using model explainability.


### Set-Up

For local installations (e.g. laptop), create a clean virtual environment using Python 3.8. Once your virtual
 environment is created, use pip to install the 
package requirements:

```bash 
pip install -r requirements.txt
```
This project also leveraged Microsoft Azure Databricks to perform data cleaning, feature engineering, and model 
training at scale. There are two *.ipynb files that require Microsoft Azure Databricks Runtime Version 10.1. All of the 
Spark code is based on v3.2.0. 

### Repository Structure
```bash
├── README.md
├── app.py
├── data
│   ├── mini_sparkify_event_data.json
│   ├── test_data_full
│   ├── test_data_sample
│   ├── train_data_full
│   └── train_data_sample
├── images
│   ├── db_screenshot.png
│   ├── sparkify.png
│   └── st_app.png
├── models
│   ├── gbt_model_sample
│   ├── sklearn_gbc_full.pkl
│   ├── sklearn_gbc_sample.pkl
│   ├── sklearn_lr_full.pkl
│   ├── sklearn_lr_sample.pkl
│   ├── spark_gbt_model_sample
│   └── spark_lr_model_sample
├── notebooks
│   ├── 1_databricks_cleaning_feature_engineering_full_data.ipynb
│   ├── 1_eda_cleaning_feature_engineering_sample.ipynb
│   ├── 2_databricks_train_eval_spark_model_full_data.ipynb
│   ├── 2_sklearn_model_training_evaluation_full.ipynb
│   ├── 2_sklearn_model_training_evaluation_sample.ipynb
│   ├── 2_spark_model_training_evaluation_sample.ipynb
│   └── 3_prediction_explainers.ipynb
├── output
│   ├── pyspark
│   │   ├── full
│   │   │   ├── GBT_Classifier_PR.png
│   │   │   ├── GBT_Classifier_ROC.png
│   │   │   ├── GBT_Classifier_test_cm.png
│   │   │   ├── GBT_Classifier_train_cm.png
│   │   │   ├── Logistic_Regression_PR.png
│   │   │   ├── Logistic_Regression_ROC.png
│   │   │   ├── Logistic_Regression_test_cm.png
│   │   │   └── Logistic_Regression_train_cm.png
│   │   └── sample
│   │       ├── GBT_PR.png
│   │       ├── GBT_ROC.png
│   │       ├── GBT_test_cm.png
│   │       ├── GBT_train_cm.png
│   │       ├── Logistic_Regression_PR.png
│   │       ├── Logistic_Regression_ROC.png
│   │       ├── Logistic_Regression_test_cm.png
│   │       └── Logistic_Regression_train_cm.png
│   ├── shap_summary.png
│   └── sklearn
│       ├── full
│       │   ├── HistGradientBoostingClassifier_PR.png
│       │   ├── HistGradientBoostingClassifier_ROC.png
│       │   ├── HistGradientBoostingClassifier_test_cm.png
│       │   ├── HistGradientBoostingClassifier_train_cm.png
│       │   ├── Logistic_Regression_PR.png
│       │   ├── Logistic_Regression_ROC.png
│       │   ├── Logistic_Regression_test_cm.png
│       │   └── Logistic_Regression_train_cm.png
│       └── sample
│           ├── HistGradientBoostingClassifier_PR.png
│           ├── HistGradientBoostingClassifier_ROC.png
│           ├── HistGradientBoostingClassifier_test_cm.png
│           ├── HistGradientBoostingClassifier_train_cm.png
│           ├── Logistic_Regression_PR.png
│           ├── Logistic_Regression_ROC.png
│           ├── Logistic_Regression_test_cm.png
│           └── Logistic_Regression_train_cm.png
├── requirements.txt
├── src
│   └── sparkifychurn
│       ├── __init__.py
│       ├── cleanData.py
│       ├── evaluateModel.py
│       ├── exploreData.py
│       ├── generateFeatures.py
│       ├── trainModel.py
│       └── utils.py
```
### File Descriptions
**app.py** - Sparkify Churn Prediction Model Dashboard built with Streamlit. 
**mini_sparkify_event_data.json** - Sample customer logs provided by Sparkify.  
**test_data_full** - Parquet files containing the full test data set.  
**test_data_sample** - Parquet files containing a sample test data set.  
**train_data_full** - Parquet files containing the full training data set.  
**train_data_sample** - Parquet files containing a sample training data set.  

**1_databricks_cleaning_feature_engineering_full_data.ipynb** - Clean data and engineer features at scale using Databricks.  
**1_eda_cleaning_feature_engineering_sample.ipynb** - Clean data and engineer features for a sample of data.  
**2_databricks_train_eval_spark_model_full_data.ipynb** - Train PySpark models on the full dataset using Databricks.   
**2_sklearn_model_training_evaluation_full.ipynb** - Train scikit-learn models on full data set.  
**2_sklearn_model_training_evaluation_sample.ipynb** - Train scikit-learn models on sample data set.  
**2_spark_model_training_evaluation_sample.ipynb** - Train PySpark models on a sample data set.  
**prediction_explainers.ipynb** - Explore top model's SHAP values.  


*Directories:*  
**output/pyspark/full** - PySpark model performance plots  on full data set.  
**output/pyspark/sample** - PySpark model performance plots on sample data.  
**output/sklearn/full** - Scikit-learn model performance plots  on full data set.  
**output/sklearn/sample** - Scikit-learn model performance plots on sample data.  
**requirements.txt** - Contains package requirements to run code.
**src/sparkifychurn** - Python package containing key functions to process data and train PySpark models. 

### EDA, Data Cleaning & Feature Engineering 
There are two notebooks associated with exploratory data analysis, cleaning, and feature engineering. I started with the
`1_eda_cleaning_feature_engineering_sample.ipynb` where I thoroughly explored a sample of the data and wrote some data 
cleaning functions available in the `sparkifychurn` package. Once I had developed code to clean and engineer the data, I 
scaled up the work using Microsoft Azure Databricks. Using a Spark cluster, I cleaned and transformed a 12 GB data set. 

### Model Training & Evaluation
There are four notebooks associated with model training and evaluation. Initial development took place using 
`2_spark_model_training_evaluation_sample.ipynb`.  Once the model training pipelines were set-up, and model evaluation 
functions were written in `sparkify`, they were scaled on Databricks in `2_databricks_train_eval_spark_model_full_data.ipynb`. 
![Databricks Application](images/db_screenshot.png). *mlflow* was used to track model performance and persisting the 
model object.  
Similar scikit-learn models were also explored for their feasibility. While the initial log file was large, transforming 
it into one-row per customer significantly reduced its dimensionality, making scikit-learn models an option. 
These two notebooks are 
 `2_sklearn_model_training_evaluation_sample.ipynb` and  `2_sklearn_model_training_evaluation_full.ipynb`. 

### Model Explainability 
After training and evaluating the models, the top scikit-learn model was used to calculate shapely
values. The overall feature importance/summary plot and an example of an individual observation's shapely values can be 
found in `3_prediction_explainers.ipynb `.

### Databricks


### Web Application
The Streamlit web application ties relevant customer attributes, churn prediction, and drivers of churn together in an 
easy to use interface. To use the web application, type in the command line in the root directory of the project: 
```bash
streamlit run app.py
```
The application should appear in your web browser at a localhost:

![Streamlit Application](images/st_app.png)
## Analysis


## Findings
Summarize end-to-end problem solution and 2 aspects found interesting or difficult
Discuss possible impro### Databricksvements. 