# === Importing Required Libraries ===
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder, OneHotEncoder

def Preprocessing(datasets):
    """
    Unsupervised data preprocessing including missing value handling,
    feature engineering, encoding, and normalization.
    """

    # === Removing Invalid Rows ===
    # Removing rows where 'StatisticGroupKod' is -1 due to a typing error in the data
    datasets = datasets[datasets['StatisticGroupKod'] != -1]

    # === Handling Missing Values ===
    # Creating mappings for missing values based on existing data
    train_mappings = {
        'StatisticArea': datasets.dropna(subset=['StatisticArea'])
                                .set_index('StatisticAreaKod')['StatisticArea'].to_dict(),
        'Yeshuv': datasets.dropna(subset=['Yeshuv'])
                          .set_index('YeshuvKod')['Yeshuv'].to_dict()
    }

    # Filling missing values based on mappings
    datasets['StatisticArea'] = datasets['StatisticArea'].fillna(
        datasets['StatisticAreaKod'].map(train_mappings['StatisticArea']))
    datasets['Yeshuv'] = datasets['Yeshuv'].fillna(
        datasets['YeshuvKod'].map(train_mappings['Yeshuv']))

    # === Data Cleaning and Column Removal ===
    # Removing rows where 'Yeshuv' is still missing
    datasets = datasets.dropna(subset=['Yeshuv'])

    # Removing columns with excessive missing values (more than 85% missing data)
    columns_to_remove_85 = ['municipalKod', 'municipalName']
    datasets = datasets.drop(columns=columns_to_remove_85, errors='ignore')

    # Removing columns that were used for mapping and are now redundant
    columns_to_remove_after_fill = ['StatisticAreaKod', 'YeshuvKod']
    datasets = datasets.drop(columns=columns_to_remove_after_fill, errors='ignore')

    # Removing additional redundant columns
    columns_to_remove_redundant = [
        'PoliceMerhavKod', 'PoliceDistrictKod', 'PoliceStationKod', 'StatisticTypeKod']
    datasets = datasets.drop(columns=columns_to_remove_redundant, errors='ignore')

    # === Handling Missing Values for 'StatisticArea' ===
    # Probabilistic filling for 'StatisticArea' based on the most frequent value within 'Yeshuv'
    def fill_statistic_area_random(yeshuv_group):
        modes = yeshuv_group.mode()
        if len(modes) > 1:
            return np.random.choice(modes)
        elif len(modes) == 1:  # תיקון תחביר כאן
            return modes.iloc[0]
        else:
            return np.nan

    # Applying the probabilistic filling method for missing 'StatisticArea' values
    datasets['StatisticArea'] = datasets.groupby('Yeshuv')['StatisticArea'].transform(
        lambda x: x.fillna(fill_statistic_area_random(x)))
    # Dropping rows where 'StatisticArea' could not be filled
    datasets = datasets.dropna(subset=['StatisticArea'])

    # === Feature Engineering and Transformation ===
    # Creating Cyclical Time Features for the Quarter (sin/cos transformation)
    datasets['Quarter_numeric'] = datasets['Quarter'].str.extract(r'(\d)').astype(int)
    datasets['Quarter_sin'] = np.sin(2 * np.pi * datasets['Quarter_numeric'] / 4)
    datasets['Quarter_cos'] = np.cos(2 * np.pi * datasets['Quarter_numeric'] / 4)

    # Calculating Crime Rate and Annual Crime Trends based on historical data
    datasets['YeshuvCrimeRate'] = datasets.groupby('Yeshuv')['Yeshuv'].transform('count')
    datasets['CrimeTrend'] = datasets.groupby('Year')['Year'].transform('count')

    # Adding interaction features: CrimeTrend multiplied by YeshuvCrimeRate
    datasets['CrimeTrend_CrimeRate'] = datasets['CrimeTrend'] * datasets['YeshuvCrimeRate']

    # Calculating average crime rate per police station
    station_crime_rate = datasets.groupby('PoliceStation')['YeshuvCrimeRate'].transform('mean')
    datasets['StationCrimeRateAvg'] = station_crime_rate

    # Calculating historical crime rate per Yeshuv
    datasets['YeshuvHistoricalCrimeRate'] = datasets.groupby('Yeshuv')['YeshuvCrimeRate'].transform('mean')

    # === Urban vs. Rural Classification ===
     # Calculating the number of nearby police stations within the same district
    datasets['StationsNearbyCount'] = datasets.groupby('PoliceDistrict')['PoliceStation'].transform('nunique')

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
    categorical_columns = ['Yeshuv', 'PoliceStation', 'StatisticArea', 'PoliceMerhav', 'PoliceDistrict', 'StatisticType', 'FictiveIDNumber']
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        datasets[col] = le.fit_transform(datasets[col])
        label_encoders[col] = le

    # === Normalization of Numeric Features ===
    numeric_columns = ['YeshuvCrimeRate', 'CrimeTrend', 'CrimeTrend_CrimeRate', 'StationCrimeRateAvg',
                       'YeshuvHistoricalCrimeRate', 'StationsNearbyCount']

    scaler = RobustScaler()
    datasets[numeric_columns] = scaler.fit_transform(datasets[numeric_columns])

    # Preventing negative values by shifting the data
    for col in numeric_columns:
        min_value = datasets[col].min()
        if min_value < 0:
            datasets[col] = datasets[col] + abs(min_value) + 1e-5

    # === Data Cleaning and Deduplication ===
    datasets = datasets.drop_duplicates()
    datasets = datasets.drop(columns=['Quarter'], errors='ignore')

    # Reducing memory usage
    def optimize_data_types(df):
        for col in df.columns:
            col_type = df[col].dtype
            if col_type == 'object':
                df[col] = df[col].astype('category')
            elif col_type == 'float64':
                df[col] = df[col].astype('float32')
            elif col_type == 'int64':
                df[col] = df[col].astype('int32')
        return df

    datasets = optimize_data_types(datasets)

    return datasets