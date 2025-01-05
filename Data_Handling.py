
#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/orifelszer/CrimeData/blob/eden-branch/Data_Handling.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


# Importing necessary libraries for data preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import glob

def Preprocessing(datasets, train_mappings=None, scaler=None, fit_scaler=False):

    # חישוב מיפויים אם אין train_mappings
    if train_mappings is None:
        train_mappings = {
            'StatisticArea': datasets.dropna(subset=['StatisticArea']).set_index('StatisticAreaKod')['StatisticArea'].to_dict(),
            'Yeshuv': datasets.dropna(subset=['Yeshuv']).set_index('YeshuvKod')['Yeshuv'].to_dict()}

    # שימוש במיפויים למילוי ערכים חסרים
    datasets['StatisticArea'] = datasets['StatisticArea'].fillna(datasets['StatisticAreaKod'].map(train_mappings['StatisticArea']))
    datasets['Yeshuv'] = datasets['Yeshuv'].fillna(datasets['YeshuvKod'].map(train_mappings['Yeshuv']))

    # מחיקת שורות עם ערכים חסרים ב"יישוב"
    datasets = datasets.dropna(subset=['Yeshuv'])

    #We will remove the columns "municipalKod" and "municipalName" because more than 85% of the data are missing values.
    columns_to_remove_85 = ['municipalKod', 'municipalName']
    datasets = datasets.drop(columns=columns_to_remove_85, errors='ignore')

    #We will remove StatisticTypeKod and 'StatisticType' because our target column is 'StatisticGroup" and 'StatisticType' belongs to 'StatisticGroup'.
    datasets = datasets.drop(columns=['StatisticTypeKod', 'StatisticType'], errors='ignore')

    #We attempted to fill the missing values in "StatisticArea" and "Yeshuv" using mappings, but this approach did not resolve the missing data.
    #Since the missing values in "StatisticArea" and "Yeshuv" could not be filled using their respective codes,
    #we decided to remove the columns "YeshuvKod" and "StatisticAreaKod" as they are no longer useful.
    columns_to_remove_after_fill = ['StatisticAreaKod', 'YeshuvKod']
    datasets = datasets.drop(columns=columns_to_remove_after_fill, errors='ignore')

    #We filled the missing values in each dataset's "StatisticArea" column by mapping the most common "StatisticArea" within each "PoliceDistrict".
    #Additionally, we added an indicator column, "is_missing_StatisticArea", to track rows where data was originally missing.

    datasets['is_missing_StatisticArea'] = datasets['StatisticArea'].isnull().astype(int)
    statistic_area_map = datasets.groupby('PoliceDistrict')['StatisticArea'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None).to_dict()
    datasets['StatisticArea'] = datasets['StatisticArea'].fillna(datasets['PoliceDistrict'].map(statistic_area_map))

    #We removed the columns "PoliceMerhavKod", "PoliceDistrictKod", and "PoliceStationKod" because they are either redundant, irrelevant to the analysis.
    columns_to_remove_redundant = ['PoliceMerhavKod', 'PoliceDistrictKod', 'PoliceStationKod', 'PoliceMerhav', 'PoliceDistrict', 'StatisticArea']
    datasets = datasets.drop(columns=columns_to_remove_redundant, errors='ignore')

    #We checked for duplicates using the 'FictiveIDNumber' column, which uniquely identifies complaints. We noticed some complaints sharing the same 'FictiveIDNumber'.
    #However, upon closer inspection, we found that these rows represent the same complainant reporting multiple charges in the StatisticGroup.
    #To handle this, we can apply one-hot encoding to the StatisticGroup column, allowing each charge to be represented as a separate feature.

    #Encoding the Data
    categorical_columns = ['Yeshuv', 'PoliceStation']
    data_encoded = pd.get_dummies(datasets, columns=categorical_columns, drop_first=True)

    #Removing the duplicaite rows
    data_encoded = data_encoded.drop_duplicates()

    #Feature Engineering
    data_encoded['Quarter_numeric'] = data_encoded['Quarter'].str.extract('(\d)').astype(int)
    data_encoded['Quarter_sin'] = np.sin(2 * np.pi * data_encoded['Quarter_numeric'] / 4)
    data_encoded['Quarter_cos'] = np.cos(2 * np.pi * data_encoded['Quarter_numeric'] / 4)

    data_encoded['YeshuvCrimeRate'] = datasets.groupby('Yeshuv')['Yeshuv'].transform('count')
    data_encoded['CrimeTrend'] = data_encoded.groupby('Year')['Year'].transform('count')

    # נרמול ערכים מספריים
    numeric_columns = ['YeshuvCrimeRate', 'CrimeTrend']
    if scaler is None:
        scaler = MinMaxScaler()
    if fit_scaler:
        data_encoded[numeric_columns] = scaler.fit_transform(data_encoded[numeric_columns])
    else:
        data_encoded[numeric_columns] = scaler.transform(data_encoded[numeric_columns])

    #Now we can remove the ID column
    data_encoded = data_encoded.drop(columns=['FictiveIDNumber','Quarter'], errors='ignore')

    return data_encoded, train_mappings, scaler

