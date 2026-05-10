# **Global Mortality Trend Analysis (Non‑Communicable Diseases)**

## **📌 Project Overview**

This project investigates global mortality trends associated with Non‑Communicable Diseases (NCDs), with a focused case study on Nepal. Using statistical modelling, clustering, and machine‑learning techniques, the project identifies high‑risk demographic groups and uncovers actionable insights to support health resource allocation and policy planning.

## **🎯 Objective**

To analyse long‑term NCD mortality trends and quantify how evolving risk factors influence mortality probability, enabling the identification of high‑risk demographic clusters.

## **🧠 Methodology**

### **1. Clustering (K‑Means)**

Used to group populations based on mortality patterns and risk factor prevalence.

Helps visualise disease progression and identify high‑risk demographic clusters.

### **2. Predictive Modelling**

Multivariate Linear Regression to quantify the impact of metabolic, behavioural, and cardiovascular risk factors.

WEKA Classification Models (Decision Tree, RandomForest, Logistic Regression) to validate risk‑factor importance and predict mortality categories.

### **3. Data Integration**

Mortality data (2000–2021) merged with IHME‑GBD risk factor prevalence (2000–2023).

Aggregated by Year and Sex to create a unified analytical dataset.

## **🛠 Tools Used**

### **Python**

Data cleaning and preprocessing

Regression modelling

Exploratory analysis

Libraries: pandas, numpy, scikit‑learn, seaborn, matplotlib

### **Tableau**

Interactive dashboards

Mortality heatmaps

Trend visualisations

Packaged workbook: .twbx

### **WEKA**

Initial classification experiments

Tree‑based modelling (e.g., **J48_tree.model**)

Model comparison and validation

## **📊 Data Sources**

### **1. Nepal NCD Mortality Dataset**

A curated secondary dataset (2000–2021) containing mortality probabilities for Nepal by sex and year.
Provided as part of coursework materials and used as the primary outcome variable.

### **2. IHME‑GBD Risk Factor Dataset**

Global Burden of Disease Study 2023

Contains prevalence of major NCD risk factors across years, sexes, and age groups.

Source: Global Burden of Disease Collaborative Network. GBD 2023 Results. Institute for Health Metrics and Evaluation (IHME), 2024.

## **🔍 Key Insights**

Metabolic risks (BMI, glucose, blood pressure) show strong upward trends and strong correlations with mortality.

Behavioural risks (tobacco, alcohol, inactivity) remain high, especially among males.

Logistic Regression achieved the highest classification accuracy (**93.9%**), followed by Decision Tree (**92.4%**) and RandomForest (**89.4%**).

Nepal demonstrates a clear link between rising metabolic risks and NCD mortality over two decades.


## **📁 Repository Structure**

/Global Mortality Trend Modeling

│

├── Correlation_Heatmap.twbx              # Tableau workbook showing correlation patterns

├── IHME-GBD_2023_DATA-dcf80fda-1.csv     # IHME-GBD risk factor dataset for Nepal

├── J48_tree.model                        # WEKA Decision Tree model

├── LogisticRegression.model              # WEKA Logistic Regression model

├── Mortality_Trends.twbx                 # Tableau dashboard visualizing mortality trends

├── RandomForest.model                    # WEKA RandomForest model

├── Risk_Factor_Trends.twbx               # Tableau dashboard visualizing risk factor trends

├── Source_(Python)_code_for_COM...       # Python script for data cleaning and regression

├── classification_data.arff              # WEKA classification dataset

└── processed_ncd_data.csv                # Cleaned and merged dataset for analysis


## **⚙️ Installation & Setup**

### **1. Clone the repository**

https://github.com/wandyboysos09/Applied-AI-Data-Science-Portfolio.git

### **2. Install Python dependencies**

_pip install -r requirements.txt_

### **3. Open Tableau dashboard**

Open the .twbx file in Tableau Desktop.

### **4. Load WEKA models**

Open WEKA → Classifier → Load Model → select .model file.

## **▶️ How to Run the Project**

### **Python Analysis**

Run Jupyter notebooks: _jupyter notebook_

### **Python Analysis**

Run the Python scripts inside the /scripts or /notebooks folder.

### **WEKA Models**

Open the .model files in WEKA to view classification results.

### **Tableau Dashboard**

Open the .twbx file to explore interactive visualisations.

## **📦 Requirements**

Python 3.9+

Tableau Desktop 2021+

WEKA 3.8+

Required Python libraries: pandas, numpy, scikit‑learn, seaborn, matplotlib

## **📌 Limitations & Future Work**

### **Limitations**

Nepal mortality dataset is a compiled secondary source without official metadata.

Risk factor data aggregated at national level — no district‑level granularity.

Models do not include forecasting (future mortality predictions).

### **Future Work**

Extend analysis to multiple countries for cross‑regional comparison.

Add time‑series forecasting (ARIMA, Prophet, LSTM).

Incorporate socio‑economic indicators (income, education, urbanisation).

## **🙏 Acknowledgements**

Institute for Health Metrics and Evaluation (IHME) for GBD data.

Solent University MSc Applied AI & Data Science programme.

Tools: Python, Tableau, WEKA.

## **📬 Contact**

### **Wande Sosina**  

MSc Applied AI & Data Science

Southampton, UK
