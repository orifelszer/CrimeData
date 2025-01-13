# **Crime Trends and Predictions: Analyzing and Classifying Criminal Records in Israel**

## **Project Overview**
This repository contains the implementation of a data mining course project focused on analyzing and predicting crime trends in Israel. Using a dataset spanning six years (2019–2024), the project employs supervised and unsupervised machine learning techniques to classify crimes, identify crime hotspots, and forecast crime rates for 2025. The ultimate goal is to use these insights to recommend optimal locations for future police stations.

## **Dataset**
The dataset includes:
- **Size:** Over 200,000 rows per file
- **Features:** 19, including:
  - **Location:** Yeshuv, Police District
  - **Time:** Year, Quarter
  - **Crime Details:** Statistic Group, Statistic Type

The dataset provides detailed temporal and spatial data on crimes in Israel. You can find them in our repository as:
- `crimes2019.zip`
- `crimes2020.zip`
- `crimes2021.zip`
- `crimes2022.zip`
- `crimes2023.zip`
- `crimes2024.zip`

## **Objectives**
- **Analyze and classify crime data:**
  - Preprocess and clean data to handle missing values and irrelevant features
  - Engineer new features, including crime rates and trends
  - Apply supervised models such as Random Forest, LightGBM, and Deep Neural Networks
- **Perform unsupervised analysis:**
  - Cluster crime hotspots using methods like K-Means and DBSCAN
  - Perform anomaly detection using Random Forest
- **Visualize results** for interpretability and insights.

## **Methodology**

### **Data Preprocessing**
- **Handling Missing Values:** 
  - Created mappings for 'StatisticArea' and 'Yeshuv' to fill missing values using their respective Kod columns.
  - Dropped rows where 'Yeshuv' is still missing after imputation.
  - Removed columns with excessive missing values (more than 85% missing).
  - Removed columns that could cause data leakage or were redundant after mapping.
  - Created a binary indicator for missing 'StatisticArea' values.
  - Applied probabilistic filling for missing 'StatisticArea' based on the most frequent value within 'Yeshuv'.
  
### **Feature Engineering**
- **Derived Features:** 
  - Created cyclical time features for the Quarter using sine and cosine transformations.
  - Calculated crime rates and trends based on historical data.
  - Added interaction features, such as CrimeTrend multiplied by YeshuvCrimeRate.
  - Calculated average crime rate per police station and historical crime rate per Yeshuv.
  - Calculated the number of nearby police stations within the same district.
- **Urban vs. Rural Classification:**
  - Classified 'Yeshuv' as 'City' or 'Moshav' using a predefined list of cities in Hebrew.

### **Supervised Learning**
- **Models Used:**
  - Random Forest
  - Decision Trees
  - LightGBM
  - Deep Neural Networks
  - XGB
- **Evaluation Metrics:** Assessed model performance using metrics such as Accuracy, Precision, Recall, and F1-score.

### **Unsupervised Learning**
- **Clustering Methods:**
  - Applied K-Means and DBSCAN to identify crime patterns and hotspots.
  - Used the Elbow Method and Silhouette Score to determine the optimal number of clusters for K-Means.
  - Visualized clusters using PCA.
- **Anomaly Detection:**
  - Used Isolation Forest and Random Forest for detecting anomalies in the data.
  - Visualized anomalies using scatter plots with Seaborn.

### **Visualization**
- **Missing Data Visualization:** Used dendrograms to visualize the dissimilarity between missing data patterns across columns.
- **Geographical Distribution:** Mapped crime data to visualize geographical patterns and hotspots.
- **Distribution Analysis:** Created histograms and boxplots to examine the distribution and detect outliers in numerical features.
- **Trend Analysis:** Visualized the top StatisticGroup categories per quarter to highlight key trends.


## **Usage**

Run the notebooks in the following order order:

 *For Supervised Models*

Since our dataset contains over two millions rows, we suggest you to call from our repository the saved cleaned datasets for `X_test_supervised.zip`, `X_train_supervised.zip`, `y_test_supervised.zip` and `y_train_supervised.zip`.
If you prefer to download yourself new versions, you can do it in the `creating_file.ipynb`

   - `Creating_supervised_data.ipynb` for data spliting
   - `Prepare_supervised_data_functions.ipynb` for data preprocessing and feature engineering
   - `Baseline_model.ipynb`, `RandomForest.ipynb`, `DecisionTree.ipynb`, `LGBM.ipynb`, `XGB.ipynb` and `DNN.ipynb` for supervised learning models
   - `Visualization.ipynb` for data visualization and supervised models results



*For Unsupervised Models* 

For the same reason, but for our unsupervised models, we suggest you to call `Clean_data_unsupervised_part1.zip` and `Clean_data_unsupervised_part2.zip`. 
You can created your datasets in the `Creating_unsupervised_data.ipynb`

   - `Creating_unsupervised_data.ipynb` for combining the datasets
   - `Prepare_unsupervised_data_functions.ipynb` for data preprocessing and feature engineering
   - `Clustering.ipynb` and `Anomaly_detectuon.ipynb` for Unsupervised learning models 


## **Installation Instructions**

Follow these steps to set up the project on your local machine:

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/orifelszer/CrimeData.git
   cd CrimeData
2. Use a virtual environment to manage your project dependencies and install the `requirements.txt` by using
   
   ```sh
    pip install -r requirements.txt

## **Contributors**
- Oriana Felszer
- Eden Shmuel

## **Acknowledgments**
This project is part of the Data Mining 2024 course. The dataset was sourced from the Israeli government open data portal of the Central Bureau of Statistics ([https://info.data.gov.il/home/](https://info.data.gov.il/home/)).

## **License**
[MIT License](LICENSE)
