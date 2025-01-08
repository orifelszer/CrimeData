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

The dataset provides detailed temporal and spatial data on crimes in Israel.

## **Objectives**
- **Build an end-to-end classification pipeline:**
  - Preprocess and clean data
  - Engineer new features, including crime rates and trends
  - Apply supervised models such as Random Forest, LightGBM, and Deep Neural Networks
- **Perform unsupervised analysis:**
  - Cluster crime hotspots using K-Means and DBSCAN
- **Visualize results** for interpretability and insights.

## **Methodology**
- **Data Preprocessing:**
  - Handled missing values and irrelevant features
  - Encoded categorical variables and normalized features
- **Feature Engineering:**
  - Derived features such as quarterly crime rates and trends
- **Supervised Learning:**
  - Random Forest, Decision Trees, LightGBM, and Deep Neural Networks
  - **Evaluation metrics:** Accuracy, Precision, Recall, F1-score
- **Unsupervised Learning:**
  - Clustering methods (K-Means, DBSCAN) to identify crime patterns
  - Anomaly Detection using Isolation Forest
- **Visualization:**
  - Confusion matrices, trend plots, and spatial distributions for better interpretability

## **Usage**
1. Run the notebooks in order:
   - `prepare_data.ipynb` for data preprocessing
   - `creating_file.ipynb` for feature engineering
   - `RandomForest.ipynb`, `DecisionTree.ipynb`, `LGBM.ipynb`, and `DNN.ipynb` for supervised learning
   - `Visualization.ipynb` for data analysis and result visualization
2. Explore the outputs in the `results/` folder.

## **Contributors**
- Oriana Felszer
- Eden Shmuel

## **Acknowledgments**
This project is part of the Data Mining 2024 course. The dataset was sourced from the Israeli government open data portal of the Central Bureau of Statistics ([https://info.data.gov.il/home/](https://info.data.gov.il/home/)).

## **License**
[MIT License](LICENSE)
