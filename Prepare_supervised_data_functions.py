from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

# === Mapping and Imputation for Missing Values ===
class FillMissingValues(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.train_mappings = None

    def fit(self, X, y=None):
        # Creating mappings for 'StatisticArea' and 'Yeshuv' to fill missing values
        if self.train_mappings is None:
            self.train_mappings = {
                'StatisticArea': X.dropna(subset=['StatisticArea'])
                                   .set_index('StatisticAreaKod')['StatisticArea'].to_dict(),
                'Yeshuv': X.dropna(subset=['Yeshuv'])
                           .set_index('YeshuvKod')['Yeshuv'].to_dict()
            }
        return self

    def transform(self, X):
        # Fill missing values using the mappings created above
        X_filled = X.copy()
        X_filled['StatisticArea'] = X_filled['StatisticArea'].fillna(
            X_filled['StatisticAreaKod'].map(self.train_mappings['StatisticArea'])
        )
        X_filled['Yeshuv'] = X_filled['Yeshuv'].fillna(
            X_filled['YeshuvKod'].map(self.train_mappings['Yeshuv'])
        )
        return X_filled

# === Data Cleaning and Column Removal ===
class DataCleaning(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Remove rows where 'Yeshuv' is still missing
        X_cleaned = X.copy()
        X_cleaned = X_cleaned.dropna(subset=['Yeshuv'])

        # Remove columns with excessive missing values (more than 85% missing)
        columns_to_remove_85 = ['municipalKod', 'municipalName']
        X_cleaned = X_cleaned.drop(columns=columns_to_remove_85, errors='ignore')

        # Remove columns that may cause data leakage
        X_cleaned = X_cleaned.drop(columns=['StatisticTypeKod', 'StatisticType'], errors='ignore')

        # Remove columns used for mapping that are now redundant
        columns_to_remove_after_fill = ['StatisticAreaKod', 'YeshuvKod']
        X_cleaned = X_cleaned.drop(columns=columns_to_remove_after_fill, errors='ignore')

        # Remove additional redundant columns
        columns_to_remove_redundant = ['PoliceMerhavKod', 'PoliceDistrictKod', 'PoliceStationKod']
        X_cleaned = X_cleaned.drop(columns=columns_to_remove_redundant, errors='ignore')

        return X_cleaned

# === Handling Missing Values for 'StatisticArea' ===
class ImputeStatisticArea(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_filled = X.copy()
        # Create a binary indicator for missing 'StatisticArea' values
        X_filled['is_missing_StatisticArea'] = X_filled['StatisticArea'].isnull().astype(int)

        # Define a helper function for probabilistic filling
        def fill_statistic_area_random(yeshuv_group):
            modes = yeshuv_group.mode()
            if len(modes) > 1:
                return np.random.choice(modes)
            elif len(modes) == 1:
                return modes.iloc[0]
            else:
                return np.nan

        # Apply probabilistic filling for missing 'StatisticArea'
        X_filled['StatisticArea'] = X_filled.groupby('Yeshuv')['StatisticArea'].transform(
            lambda x: x.fillna(fill_statistic_area_random(x)))

        # Drop rows where 'StatisticArea' could not be filled
        X_filled = X_filled.dropna(subset=['StatisticArea'])

        return X_filled

# === Feature Engineering and Transformation ===
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_fe = X.copy()

        # Creating Cyclical Time Features for the Quarter (sin/cos transformation)
        X_fe['Quarter_numeric'] = X_fe['Quarter'].str.extract(r'(\d)').astype(int)
        X_fe['Quarter_sin'] = np.sin(2 * np.pi * X_fe['Quarter_numeric'] / 4)
        X_fe['Quarter_cos'] = np.cos(2 * np.pi * X_fe['Quarter_numeric'] / 4)

        # Calculating Crime Rates and Annual Trends
        X_fe['YeshuvCrimeRate'] = X_fe.groupby('Yeshuv')['Yeshuv'].transform('count')
        X_fe['CrimeTrend'] = X_fe.groupby('Year')['Year'].transform('count')

        # Adding Interaction Features
        X_fe['CrimeTrend_CrimeRate'] = X_fe['CrimeTrend'] * X_fe['YeshuvCrimeRate']

        # Calculating Average Crime Rates
        X_fe['StationCrimeRateAvg'] = X_fe.groupby('PoliceStation')['YeshuvCrimeRate'].transform('mean')
        X_fe['YeshuvHistoricalCrimeRate'] = X_fe.groupby('Yeshuv')['YeshuvCrimeRate'].transform('mean')

        # Counting Nearby Police Stations
        X_fe['StationsNearbyCount'] = X_fe.groupby('PoliceDistrict')['PoliceStation'].transform('nunique')

        return X_fe

# === Urban vs. Rural Classification ===
class UrbanRuralClassification(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_classified = X.copy()

        # Defining the list of Hebrew cities
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
        X_classified['CityOrMoshav'] = X_classified['Yeshuv'].apply(
            lambda x: 'City' if x in hebrew_cities else 'Moshav')

        return X_classified

# === Encoding Categorical Variables and Normalizing Numeric Features ===
class EncodingAndScaling(BaseEstimator, TransformerMixin):
    def __init__(self, fit_scaler=True):
        self.label_encoders = {}
        self.scaler = RobustScaler()
        self.fit_scaler = fit_scaler

    def fit(self, X, y=None):
        # Applying Label Encoding for categorical columns
        self.categorical_columns = ['Yeshuv', 'PoliceStation', 'StatisticArea', 'PoliceMerhav', 'PoliceDistrict']
        self.numeric_columns = ['YeshuvCrimeRate', 'CrimeTrend', 'CrimeTrend_CrimeRate',
                                'StationCrimeRateAvg', 'YeshuvHistoricalCrimeRate', 'StationsNearbyCount']

        for col in self.categorical_columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.label_encoders[col] = le

        # Fitting the scaler only if required
        if self.fit_scaler:
            self.scaler.fit(X[self.numeric_columns])

        return self

    def transform(self, X):
        X_transformed = X.copy()

        # === Applying One-Hot Encoding to 'CityOrMoshav' ===
        X_transformed = pd.get_dummies(X_transformed, columns=['CityOrMoshav'], drop_first=True)

        # === Applying Label Encoding to categorical columns ===
        for col, le in self.label_encoders.items():
            X_transformed[col] = le.transform(X_transformed[col])

        # === Normalization of Numeric Features ===
        if self.fit_scaler:
            X_transformed[self.numeric_columns] = self.scaler.transform(X_transformed[self.numeric_columns])

        # Preventing negative values by adding a small positive constant
        for col in self.numeric_columns:
            min_value = X_transformed[col].min()
            if min_value < 0:
                X_transformed[col] = X_transformed[col] + abs(min_value) + 1e-5

        return X_transformed

# === Memory Usage Reduction ===
class MemoryReduction(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_optimized = X.copy()

        # Optimize data types to reduce memory usage
        for col in X_optimized.columns:
            col_type = X_optimized[col].dtype
            if col_type == 'object':
                X_optimized[col] = X_optimized[col].astype('category')
            elif col_type == 'float64':
                X_optimized[col] = X_optimized[col].astype('float32')
            elif col_type == 'int64':
                X_optimized[col] = X_optimized[col].astype('int32')

        return X_optimized

# === Data Cleaning and Deduplication ===
class DataCleaningAndDeduplication(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_cleaned = X.copy()

        # Removing duplicate rows
        X_cleaned = X_cleaned.drop_duplicates()

        # Removing specific columns that are no longer needed
        X_cleaned = X_cleaned.drop(columns=['FictiveIDNumber', 'Quarter'], errors='ignore')

        return X_cleaned

from sklearn.pipeline import Pipeline

# === Creating the full preprocessing pipeline ===
pipeline = Pipeline([
    ('fill_missing', FillMissingValues()),
    ('feature_engineering', FeatureEngineering()),
    ('urban_rural_classification', UrbanRuralClassification()),
    ('encoding_scaling', EncodingAndScaling()),
    ('memory_reduction', MemoryReduction()),
    ('data_cleaning', DataCleaningAndDeduplication())
])

# # === Importing Required Libraries ===
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import RobustScaler, LabelEncoder, OneHotEncoder

# def Preprocessing(datasets, train_mappings=None, scaler=None, fit_scaler=False):
#     """
#     Function for preprocessing datasets including missing value handling,
#     feature engineering, and label encoding.
#     """

#     # === Mapping and Imputation for Missing Values ===
#     # Creating mappings for 'StatisticArea' and 'Yeshuv' to fill missing values
#     if train_mappings is None:
#         train_mappings = {
#             'StatisticArea': datasets.dropna(subset=['StatisticArea'])
#                                     .set_index('StatisticAreaKod')['StatisticArea'].to_dict(),
#             'Yeshuv': datasets.dropna(subset=['Yeshuv'])
#                               .set_index('YeshuvKod')['Yeshuv'].to_dict()
#         }

#     # Fill missing values using the mappings created above
#     datasets['StatisticArea'] = datasets['StatisticArea'].fillna(
#         datasets['StatisticAreaKod'].map(train_mappings['StatisticArea']))
#     datasets['Yeshuv'] = datasets['Yeshuv'].fillna(
#         datasets['YeshuvKod'].map(train_mappings['Yeshuv']))

#     # === Data Cleaning and Column Removal ===
#     # Remove rows where 'Yeshuv' is still missing
#     datasets = datasets.dropna(subset=['Yeshuv'])

#     # Remove columns with excessive missing values (more than 85% missing)
#     columns_to_remove_85 = ['municipalKod', 'municipalName']
#     datasets = datasets.drop(columns=columns_to_remove_85, errors='ignore')

#     # Remove columns that may cause data leakage as they are subcategories of the target variable
#     datasets = datasets.drop(columns=['StatisticTypeKod', 'StatisticType'], errors='ignore')

#     # Remove columns that duplicate information already stored in text format
#     # These columns ('StatisticAreaKod' and 'YeshuvKod') were previously used for mapping and are now redundant
#     columns_to_remove_after_fill = ['StatisticAreaKod', 'YeshuvKod']
#     datasets = datasets.drop(columns=columns_to_remove_after_fill, errors='ignore')

#     # Remove additional redundant columns related to police districts and areas
#     columns_to_remove_redundant = [
#         'PoliceMerhavKod', 'PoliceDistrictKod', 'PoliceStationKod']
#     datasets = datasets.drop(columns=columns_to_remove_redundant, errors='ignore')

#     # === Handling Missing Values for 'StatisticArea' ===
#     # Create a binary indicator for missing 'StatisticArea' values
#     datasets['is_missing_StatisticArea'] = datasets['StatisticArea'].isnull().astype(int)

#     # Probabilistic filling for 'StatisticArea' based on the most frequent value within 'Yeshuv'
#     def fill_statistic_area_random(yeshuv_group):
#         modes = yeshuv_group.mode()
#         if len(modes) > 1:
#             return np.random.choice(modes)
#         elif len(modes) == 1:  # תיקון תחביר כאן
#             return modes.iloc[0]
#         else:
#             return np.nan

#     # Apply probabilistic filling for missing 'StatisticArea'
#     datasets['StatisticArea'] = datasets.groupby('Yeshuv')['StatisticArea'].transform(
#         lambda x: x.fillna(fill_statistic_area_random(x)))
#     datasets = datasets.dropna(subset=['StatisticArea'])

#     # === Feature Engineering and Transformation ===
#     # Creating Cyclical Time Features for the Quarter (sin/cos transformation)
#     datasets['Quarter_numeric'] = datasets['Quarter'].str.extract(r'(\d)').astype(int)
#     datasets['Quarter_sin'] = np.sin(2 * np.pi * datasets['Quarter_numeric'] / 4)
#     datasets['Quarter_cos'] = np.cos(2 * np.pi * datasets['Quarter_numeric'] / 4)

#     # Calculating Crime Rate and Annual Crime Trends based on historical data
#     datasets['YeshuvCrimeRate'] = datasets.groupby('Yeshuv')['Yeshuv'].transform('count')
#     datasets['CrimeTrend'] = datasets.groupby('Year')['Year'].transform('count')

#     # Adding interaction features: CrimeTrend multiplied by YeshuvCrimeRate
#     datasets['CrimeTrend_CrimeRate'] = datasets['CrimeTrend'] * datasets['YeshuvCrimeRate']

#     # Calculating average crime rate per police station
#     station_crime_rate = datasets.groupby('PoliceStation')['YeshuvCrimeRate'].transform('mean')
#     datasets['StationCrimeRateAvg'] = station_crime_rate

#     # Calculating historical crime rate per Yeshuv
#     datasets['YeshuvHistoricalCrimeRate'] = datasets.groupby('Yeshuv')['YeshuvCrimeRate'].transform('mean')

#     # Calculating the number of nearby police stations within the same district
#     datasets['StationsNearbyCount'] = datasets.groupby('PoliceDistrict')['PoliceStation'].transform('nunique')

#     # === Urban vs. Rural Classification ===
#     # Defining a list of cities in Hebrew for classification purposes
#     hebrew_cities = [
#         "אום אל-פחם", "אופקים", "אור יהודה", "אור עקיבא", "אילת", "אלעד", "אריאל",
#         "אשדוד", "אשקלון", "באקה אל-גרביה", "באר שבע", "בית שאן", "בית שמש", "ביתר עילית",
#         "בני ברק", "בת ים", "גבעת שמואל", "גבעתיים", "דימונה", "הוד השרון", "הרצלייה",
#         "חדרה", "חולון", "חיפה", "טבריה", "טייבה", "טירה", "טירת כרמל", "טמרה",
#         "יבנה", "יהוד", "יקנעם עילית", "ירושלים", "כפר יונה", "כפר סבא", "כפר קאסם",
#         "כרמיאל", "לוד", "מגדל העמק", "מודיעין-מכבים-רעות", "מודיעין עילית", "מעלה אדומים",
#         "מעלות-תרשיחא", "נהרייה", "נס ציונה", "נצרת", "נצרת עילית", "נשר", "נתיבות",
#         "נתניה", "סח'נין", "עכו", "עפולה", "עראבה", "ערד", "פתח תקווה", "צפת",
#         "קלנסווה", "קריית אונו", "קריית אתא", "קריית ביאליק", "קריית גת", "קריית ים",
#         "קריית מוצקין", "קריית מלאכי", "קריית שמונה", "ראש העין", "ראשון לציון",
#         "רהט", "רחובות", "רמלה", "רמת גן", "רמת השרון", "רעננה", "שדרות", "שפרעם",
#         "תל אביב - יפו"
#     ]

#     # Creating a new column classifying Yeshuv as 'City' or 'Moshav'
#     datasets['CityOrMoshav'] = datasets['Yeshuv'].apply(lambda x: 'City' if x in hebrew_cities else 'Moshav')

#     # === Encoding Categorical Variables ===
#     # Applying One-Hot Encoding to 'CityOrMoshav'
#     datasets = pd.get_dummies(datasets, columns=['CityOrMoshav'], drop_first=True)

#     # Applying Label Encoding to categorical columns
#     categorical_columns = ['Yeshuv', 'PoliceStation', 'StatisticArea', 'PoliceMerhav', 'PoliceDistrict']
#     label_encoders = {}
#     for col in categorical_columns:
#         le = LabelEncoder()
#         datasets[col] = le.fit_transform(datasets[col])
#         label_encoders[col] = le

#     # === Normalization of Numeric Features ===
#     numeric_columns = ['YeshuvCrimeRate', 'CrimeTrend', 'CrimeTrend_CrimeRate', 'StationCrimeRateAvg',
#                        'YeshuvHistoricalCrimeRate', 'StationsNearbyCount']

#     if scaler is None:
#         scaler = RobustScaler()
#     if fit_scaler:
#         datasets[numeric_columns] = scaler.fit_transform(datasets[numeric_columns])
#     else:
#         datasets[numeric_columns] = scaler.transform(datasets[numeric_columns])

#     # Prevent negative values by adding a small positive constant
#     for col in numeric_columns:
#         min_value = datasets[col].min()
#         if min_value < 0:
#             datasets[col] = datasets[col] + abs(min_value) + 1e-5

#     # === Data Cleaning and Deduplication ===
#     datasets = datasets.drop_duplicates()
#     datasets = datasets.drop(columns=['FictiveIDNumber', 'Quarter'], errors='ignore')

#         # === Memory Usage Reduction ===
#     def optimize_data_types(df):
#         for col in df.columns:
#             col_type = df[col].dtype
#             if col_type == 'object':
#                 df[col] = df[col].astype('category')
#             elif col_type == 'float64':
#                 df[col] = df[col].astype('float32')
#             elif col_type == 'int64':
#                 df[col] = df[col].astype('int32')
#         return df

#     datasets = optimize_data_types(datasets)

#     return datasets, train_mappings, scaler, label_encoders