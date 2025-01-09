#Importing necessary libraries for data preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder

def Preprocessing(datasets, train_mappings=None, scaler=None, fit_scaler=False):
    """
    Function for preprocessing datasets including missing value handling,
    feature engineering, and label encoding.
    """

    # === Mapping and Imputation for Missing Values ===
    # Creating mappings for 'StatisticArea' and 'Yeshuv' to fill missing values
    if train_mappings is None:
        train_mappings = {
            'StatisticArea': datasets.dropna(subset=['StatisticArea'])
                                    .set_index('StatisticAreaKod')['StatisticArea'].to_dict(),
            'Yeshuv': datasets.dropna(subset=['Yeshuv'])
                              .set_index('YeshuvKod')['Yeshuv'].to_dict()
        }

    # Fill missing values using the mappings created above
    datasets['StatisticArea'] = datasets['StatisticArea'].fillna(
        datasets['StatisticAreaKod'].map(train_mappings['StatisticArea']))
    datasets['Yeshuv'] = datasets['Yeshuv'].fillna(
        datasets['YeshuvKod'].map(train_mappings['Yeshuv']))

    # === Data Cleaning and Column Removal ===
    # Remove rows where 'Yeshuv' is still missing
    datasets = datasets.dropna(subset=['Yeshuv'])

    # Remove columns with excessive missing values (more than 85% missing)
    columns_to_remove_85 = ['municipalKod', 'municipalName']
    datasets = datasets.drop(columns=columns_to_remove_85, errors='ignore')

    # Remove columns that may cause data leakage as they are subcategories of the target variable
    datasets = datasets.drop(columns=['StatisticTypeKod', 'StatisticType'], errors='ignore')

    # Remove columns that duplicate information already stored in text format
    # These columns ('StatisticAreaKod' and 'YeshuvKod') were previously used for mapping and are now redundant
    columns_to_remove_after_fill = ['StatisticAreaKod', 'YeshuvKod']
    datasets = datasets.drop(columns=columns_to_remove_after_fill, errors='ignore')

    # Remove additional redundant columns related to police districts and areas
    columns_to_remove_redundant = [
        'PoliceMerhavKod', 'PoliceDistrictKod', 'PoliceStationKod']
    datasets = datasets.drop(columns=columns_to_remove_redundant, errors='ignore')

    # === Handling Missing Values for 'StatisticArea' ===
    # Create a binary indicator for missing 'StatisticArea' values
    datasets['is_missing_StatisticArea'] = datasets['StatisticArea'].isnull().astype(int)

    # Probabilistic filling for 'StatisticArea' based on the most frequent value within 'Yeshuv'
    def fill_statistic_area_random(yeshuv_group):
        modes = yeshuv_group.mode()
        if len(modes) > 1:
            return np.random.choice(modes)
        else:
            return modes.iloc[0]

    # Apply probabilistic filling for missing 'StatisticArea'
    datasets['StatisticArea'] = datasets.groupby('Yeshuv')['StatisticArea'].transform(
        lambda x: x.fillna(fill_statistic_area_random(x)))

    # === Feature Engineering and Transformation ===
    # Creating Cyclical Time Features for the Quarter (sin/cos transformation)
    datasets['Quarter_numeric'] = datasets['Quarter'].str.extract(r'(\d)').astype(int)
    datasets['Quarter_sin'] = np.sin(2 * np.pi * datasets['Quarter_numeric'] / 4)
    datasets['Quarter_cos'] = np.cos(2 * np.pi * datasets['Quarter_numeric'] / 4)

    # Calculating Crime Rate and Annual Crime Trends based on historical data
    datasets['YeshuvCrimeRate'] = datasets.groupby('Yeshuv')['Yeshuv'].transform('count')
    datasets['CrimeTrend'] = datasets.groupby('Year')['Year'].transform('count')

    # Adding a combined time index and a seasonal feature (Spring/Summer = 0, Autumn/Winter = 1)
    datasets['TimeIndex'] = datasets['Year'] * 4 + datasets['Quarter_numeric']
    datasets['Season'] = datasets['Quarter_numeric'].apply(lambda q: (q % 4) // 2)

    # Adding interaction features: CrimeTrend multiplied by YeshuvCrimeRate
    datasets['CrimeTrend_CrimeRate'] = datasets['CrimeTrend'] * datasets['YeshuvCrimeRate']

    # Calculating average crime rate per police station
    station_crime_rate = datasets.groupby('PoliceStation')['YeshuvCrimeRate'].transform('mean')
    datasets['StationCrimeRateAvg'] = station_crime_rate

    # Calculating number of crimes reported at each police station
    datasets['PoliceStationCrimeCount'] = datasets.groupby('PoliceStation')['PoliceStation'].transform('count')

    # Calculating historical crime rate per Yeshuv
    datasets['YeshuvHistoricalCrimeRate'] = datasets.groupby('Yeshuv')['YeshuvCrimeRate'].transform('mean')

    # Calculating the number of nearby police stations within the same district
    datasets['StationsNearbyCount'] = datasets.groupby('PoliceDistrict')['PoliceStation'].transform('nunique')

    # === Urban vs. Rural Classification ===
    # Defining a list of cities in Hebrew for classification purposes
    hebrew_cities = [
        "אום אל-פחם", "אופקים", "אור יהודה", "אור עקיבא", "אילת", "אלעד", "אריאל",
        "אשדוד", "אשקלון", "באקה אל-גרביה", "באר שבע", "בית שאן", "בית שמש", "ביתר עילית",
        "בני ברק", "בת ים", "גבעת שמואל", "גבעתיים", "דימונה", "הוד השרון", "הרצלייה",
        "חדרה", "חולון", "חיפה", "טבריה", "טייבה", "טירה", "טירת כרמל", "טמרה",
        "יבנה", "יהוד", "יקנעם עילית", "ירושלים", "כפר יונה", "כפר סבא", "כפר קאסם",
        "כרמיאל", "לוד", "מגדל העמק", "מודיעין-מכבים-רעות", "מודיעין עילית", "מעלה אדומים",
        "מעלות-תרשיחא", "נהרייה", "נס ציונה", "נצרת", "נצרת עילית", "נשר", "נתיבות",
        "נתניה", "סח'נין", "עכו", "עפולה", "עראבה", "ערד", "פתח תקווה", "צפת",
        "קלנסווה", "קריית אונו", "קריית אתא", "קריית ביאליק", "קריית גת", "קריית ים",
        "קריית מוצקין", "קריית מלאכי", "קריית שמונה", "ראש העין", "ראשון לציון",
        "רהט", "רחובות", "רמלה", "רמת גן", "רמת השרון", "רעננה", "שדרות", "שפרעם",
        "תל אביב - יפו"
    ]

    # Creating a new column classifying Yeshuv as 'City' or 'Moshav'
    datasets['CityOrMoshav'] = datasets['Yeshuv'].apply(lambda x: 'City' if x in hebrew_cities else 'Moshav')

    # === Encoding Categorical Variables ===
    # Applying One-Hot Encoding to 'CityOrMoshav'
    datasets = pd.get_dummies(datasets, columns=['CityOrMoshav'], drop_first=True)

    # Applying Label Encoding to categorical columns
    categorical_columns = ['Yeshuv', 'PoliceStation', 'StatisticArea', 'PoliceMerhav', 'PoliceDistrict']
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        datasets[col] = le.fit_transform(datasets[col])
        label_encoders[col] = le

    # === Normalization of Numeric Features ===
    numeric_columns = ['YeshuvCrimeRate', 'CrimeTrend', 'TimeIndex', 'CrimeTrend_CrimeRate', 'StationCrimeRateAvg',
                       'PoliceStationCrimeCount', 'YeshuvHistoricalCrimeRate', 'StationsNearbyCount']

    if scaler is None:
        scaler = RobustScaler()
    if fit_scaler:
        datasets[numeric_columns] = scaler.fit_transform(datasets[numeric_columns])
    else:
        datasets[numeric_columns] = scaler.transform(datasets[numeric_columns])

    # Prevent negative values by adding a small positive constant
    for col in numeric_columns:
        min_value = datasets[col].min()
        if min_value < 0:
            datasets[col] = datasets[col] + abs(min_value) + 1e-5

    # === Data Cleaning and Deduplication ===
    datasets = datasets.drop_duplicates()
    datasets = datasets.drop(columns=['FictiveIDNumber', 'Quarter'], errors='ignore')

    return datasets, train_mappings, scaler, label_encoders

