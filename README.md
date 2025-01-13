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
- **Build an end-to-end classification pipeline:**
  - Preprocess and clean data
  - Engineer new features, including crime rates and trends
  - Apply supervised models such as Random Forest, LightGBM, and Deep Neural Networks
- **Perform unsupervised analysis:**
  - Cluster crime hotspots using 
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

Run the notebooks in the following order order:

## *For Supervised Models*

Since our dataset contains over two millions rows, we suggest you to call from our repository the saved cleaned datasets for `X_test_supervised.zip`, `X_train_supervised.zip`, `y_test_supervised.zip` and `y_train_supervised.zip`.
If you prefer to download yourself new versions, you can do it in the `creating_file.ipynb`

   - `Creating_supervised_data.ipynb` for data spliting
   - `Prepare_supervised_data_functions.ipynb` for data preprocessing and feature engineering
   - `Baseline_model.ipynb`, `RandomForest.ipynb`, `DecisionTree.ipynb`, `LGBM.ipynb`, `XGB.ipynb` and `DNN.ipynb` for supervised learning models
   - `Visualization.ipynb` for data visualization and supervised models results


For the same reason, but for our unsupervised models, we suggest you to call `Clean_data_unsupervised_part1.zip` and `Clean_data_unsupervised_part2.zip`. 
You can created your datasets in the `Creating_unsupervised_data.ipynb`

## *For Unsupervised Models* 
   - `Creating_unsupervised_data.ipynb` for combining the datasets
   - `Prepare_unsupervised_data_functions.ipynb` for data preprocessing and feature engineering
   - `Clustering.ipynb` and `Anomaly_detectuon.ipynb` for Unsupervised learning models 

   


## **Contributors**
- Oriana Felszer
- Eden Shmuel

## **Acknowledgments**
This project is part of the Data Mining 2024 course. The dataset was sourced from the Israeli government open data portal of the Central Bureau of Statistics ([https://info.data.gov.il/home/](https://info.data.gov.il/home/)).

## **License**
[MIT License](LICENSE)
