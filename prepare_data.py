#Importing necessary libraries for data preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder

def Preprocessing(datasets, train_mappings=None, scaler=None, fit_scaler=False):
    """
    Function for preprocessing datasets including missing value handling,
    feature engineering, and label encoding.
    """

    # Generate mappings for 'StatisticArea' and 'Yeshuv' if not provided
    if train_mappings is None:
        train_mappings = {
            'StatisticArea': datasets.dropna(subset=['StatisticArea'])
                                    .set_index('StatisticAreaKod')['StatisticArea'].to_dict(),
            'Yeshuv': datasets.dropna(subset=['Yeshuv'])
                              .set_index('YeshuvKod')['Yeshuv'].to_dict()
        }

    # Fill missing values using the mappings
    datasets['StatisticArea'] = datasets['StatisticArea'].fillna(
        datasets['StatisticAreaKod'].map(train_mappings['StatisticArea']))
    datasets['Yeshuv'] = datasets['Yeshuv'].fillna(
        datasets['YeshuvKod'].map(train_mappings['Yeshuv']))

    # Remove rows where 'Yeshuv' is still missing
    datasets = datasets.dropna(subset=['Yeshuv'])

    # Drop columns with over 85% missing values
    columns_to_remove_85 = ['municipalKod', 'municipalName']
    datasets = datasets.drop(columns=columns_to_remove_85, errors='ignore')

    # Drop redundant columns that do not contribute to the analysis
    datasets = datasets.drop(columns=['StatisticTypeKod', 'StatisticType'], errors='ignore')

    # Remove additional columns no longer useful after filling missing values
    columns_to_remove_after_fill = ['StatisticAreaKod', 'YeshuvKod']
    datasets = datasets.drop(columns=columns_to_remove_after_fill, errors='ignore')

    # Create a binary indicator for missing 'StatisticArea' values
    datasets['is_missing_StatisticArea'] = datasets['StatisticArea'].isnull().astype(int)

    # Fill remaining missing 'StatisticArea' using the most common value within 'PoliceDistrict'
    statistic_area_map = datasets.groupby('PoliceDistrict')['StatisticArea'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None).to_dict()
    datasets['StatisticArea'] = datasets['StatisticArea'].fillna(
        datasets['PoliceDistrict'].map(statistic_area_map))

    # Remove additional redundant columns related to police districts and areas
    columns_to_remove_redundant = [
        'PoliceMerhavKod', 'PoliceDistrictKod', 'PoliceStationKod',
        'PoliceMerhav', 'PoliceDistrict'
        ]
    datasets = datasets.drop(columns=columns_to_remove_redundant, errors='ignore')

    # Convert categorical columns to numeric using Label Encoding
    categorical_columns = ['Yeshuv', 'PoliceStation', 'StatisticArea']
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        datasets[col] = le.fit_transform(datasets[col])
        label_encoders[col] = le

    # Remove duplicate rows after encoding
    datasets = datasets.drop_duplicates()

    # Feature Engineering: Extract numeric quarter and create cyclical features (sin/cos)
    datasets['Quarter_numeric'] = datasets['Quarter'].str.extract(r'(\d)').astype(int)
    datasets['Quarter_sin'] = np.sin(2 * np.pi * datasets['Quarter_numeric'] / 4)
    datasets['Quarter_cos'] = np.cos(2 * np.pi * datasets['Quarter_numeric'] / 4)

    # Create features for crime rate per Yeshuv and annual crime trend
    datasets['YeshuvCrimeRate'] = datasets.groupby('Yeshuv')['Yeshuv'].transform('count')
    datasets['CrimeTrend'] = datasets.groupby('Year')['Year'].transform('count')

    # Normalize numeric columns using MinMaxScaler
    numeric_columns = ['YeshuvCrimeRate', 'CrimeTrend']
    # יצירת סקיילר במידת הצורך
    if scaler is None:
        scaler = RobustScaler()
    # אימון הסקיילר על נתוני האימון בלבד
    if fit_scaler:
        datasets[numeric_columns] = scaler.fit_transform(datasets[numeric_columns])
    else:
        # החלת הסקיילר על נתוני הבדיקה
        datasets[numeric_columns] = scaler.transform(datasets[numeric_columns])

    # מניעת ערכים שליליים בעזרת הוספת קבוע קטן
    for col in numeric_columns:
        min_value = datasets[col].min()
        if min_value < 0:
            datasets[col] = datasets[col] + abs(min_value) + 1e-5

    # Remove the ID column and the original quarter column
    datasets = datasets.drop(columns=['FictiveIDNumber', 'Quarter'], errors='ignore')

    # Return the preprocessed dataset along with the mappings and scaler for later use
    return datasets, train_mappings, scaler, label_encoders


