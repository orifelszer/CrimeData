# Cell Start
# Importing necessary libraries for data preprocessing

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler



def Preprocessing_Multitask_Updated(datasets, train_mappings=None, scaler=None, fit_scaler=False):

    #We will remove the columns "municipalKod" and "municipalName" because more than 85% of the data are missing values.

    columns_to_remove_85 = ['municipalKod', 'municipalName']

    datasets = datasets.drop(columns=columns_to_remove_85, errors='ignore')



    #We will remove StatisticTypeKod and 'StatisticType' because our target column is 'StatisticGroup" and 'StatisticType' belongs to 'StatisticGroup'.

    datasets = datasets.drop(columns=['StatisticTypeKod', 'StatisticType'], errors='ignore')

    #We will remove 'StatisticAreaKod' and 'StatisticArea' because our target column is 'Yeshuv" and 'StatisticAreaKod' belongs to 'Yeshuv'.

    datasets = datasets.drop(columns=['StatisticAreaKod', 'StatisticArea'], errors='ignore')



    #We removed the columns "PoliceMerhavKod", "PoliceDistrictKod", and "PoliceStationKod" because they are either redundant, irrelevant to the analysis.

    columns_to_remove_redundant = ['PoliceMerhavKod', 'PoliceDistrictKod', 'PoliceStationKod', 'PoliceMerhav', 'PoliceDistrict', 'StatisticArea']

    datasets = datasets.drop(columns=columns_to_remove_redundant, errors='ignore')



    #We checked for duplicates using the 'FictiveIDNumber' column, which uniquely identifies complaints. We noticed some complaints sharing the same 'FictiveIDNumber'.

    #However, upon closer inspection, we found that these rows represent the same complainant reporting multiple charges in the StatisticGroup.

    #To handle this, we can apply one-hot encoding to the StatisticGroup column, allowing each charge to be represented as a separate feature.



    #Encoding the Data

    categorical_columns = ['PoliceStation']

    data_encoded = pd.get_dummies(datasets, columns=categorical_columns, drop_first=True)



    #Removing the duplicaite rows

    data_encoded = data_encoded.drop_duplicates()



    #Feature Engineering

    data_encoded['Quarter_numeric'] = data_encoded['Quarter'].str.extract('(\d)').astype(int)

    data_encoded['Quarter_sin'] = np.sin(2 * np.pi * data_encoded['Quarter_numeric'] / 4)

    data_encoded['Quarter_cos'] = np.cos(2 * np.pi * data_encoded['Quarter_numeric'] / 4)



    data_encoded['CrimeTrend'] = data_encoded.groupby('Year')['Year'].transform('count')



    # נרמול ערכים מספריים

    numeric_columns = ['CrimeTrend']

    if scaler is None:

        scaler = MinMaxScaler()

    if fit_scaler:

        data_encoded[numeric_columns] = scaler.fit_transform(data_encoded[numeric_columns])

    else:

        data_encoded[numeric_columns] = scaler.transform(data_encoded[numeric_columns])



    #Now we can remove the ID column

    data_encoded = data_encoded.drop(columns=['FictiveIDNumber','Quarter'], errors='ignore')



    return data_encoded, train_mappings, scaler
# Cell End

