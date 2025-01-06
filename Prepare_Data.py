{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "view-in-github"
   },
   "source": [
    "<a href=\"https://colab.research.google.com/github/orifelszer/CrimeData/blob/eden-branch/Prepare_Data.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "id": "AnWUf3rM4L9n"
   },
   "outputs": [],
   "source": [
    "# Importing necessary libraries for data preprocessing\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler\n",
    "\n",
    "def Preprocessing(datasets, train_mappings=None, scaler=None, fit_scaler=False):\n",
    "\n",
    "    # חישוב מיפויים אם אין train_mappings\n",
    "    if train_mappings is None:\n",
    "        train_mappings = {\n",
    "            'StatisticArea': datasets.dropna(subset=['StatisticArea']).set_index('StatisticAreaKod')['StatisticArea'].to_dict(),\n",
    "            'Yeshuv': datasets.dropna(subset=['Yeshuv']).set_index('YeshuvKod')['Yeshuv'].to_dict()}\n",
    "\n",
    "    # שימוש במיפויים למילוי ערכים חסרים\n",
    "    datasets['StatisticArea'] = datasets['StatisticArea'].fillna(datasets['StatisticAreaKod'].map(train_mappings['StatisticArea']))\n",
    "    datasets['Yeshuv'] = datasets['Yeshuv'].fillna(datasets['YeshuvKod'].map(train_mappings['Yeshuv']))\n",
    "\n",
    "    # מחיקת שורות עם ערכים חסרים ב\"יישוב\"\n",
    "    datasets = datasets.dropna(subset=['Yeshuv'])\n",
    "\n",
    "    #We will remove the columns \"municipalKod\" and \"municipalName\" because more than 85% of the data are missing values.\n",
    "    columns_to_remove_85 = ['municipalKod', 'municipalName']\n",
    "    datasets = datasets.drop(columns=columns_to_remove_85, errors='ignore')\n",
    "\n",
    "    #We will remove StatisticTypeKod and 'StatisticType' because our target column is 'StatisticGroup\" and 'StatisticType' belongs to 'StatisticGroup'.\n",
    "    datasets = datasets.drop(columns=['StatisticTypeKod', 'StatisticType'], errors='ignore')\n",
    "\n",
    "    #We attempted to fill the missing values in \"StatisticArea\" and \"Yeshuv\" using mappings, but this approach did not resolve the missing data.\n",
    "    #Since the missing values in \"StatisticArea\" and \"Yeshuv\" could not be filled using their respective codes,\n",
    "    #we decided to remove the columns \"YeshuvKod\" and \"StatisticAreaKod\" as they are no longer useful.\n",
    "    columns_to_remove_after_fill = ['StatisticAreaKod', 'YeshuvKod']\n",
    "    datasets = datasets.drop(columns=columns_to_remove_after_fill, errors='ignore')\n",
    "\n",
    "    #We filled the missing values in each dataset's \"StatisticArea\" column by mapping the most common \"StatisticArea\" within each \"PoliceDistrict\".\n",
    "    #Additionally, we added an indicator column, \"is_missing_StatisticArea\", to track rows where data was originally missing.\n",
    "\n",
    "    datasets['is_missing_StatisticArea'] = datasets['StatisticArea'].isnull().astype(int)\n",
    "    statistic_area_map = datasets.groupby('PoliceDistrict')['StatisticArea'].agg(\n",
    "        lambda x: x.mode().iloc[0] if not x.mode().empty else None).to_dict()\n",
    "    datasets['StatisticArea'] = datasets['StatisticArea'].fillna(datasets['PoliceDistrict'].map(statistic_area_map))\n",
    "\n",
    "    #We removed the columns \"PoliceMerhavKod\", \"PoliceDistrictKod\", and \"PoliceStationKod\" because they are either redundant, irrelevant to the analysis.\n",
    "    columns_to_remove_redundant = ['PoliceMerhavKod', 'PoliceDistrictKod', 'PoliceStationKod', 'PoliceMerhav', 'PoliceDistrict', 'StatisticArea']\n",
    "    datasets = datasets.drop(columns=columns_to_remove_redundant, errors='ignore')\n",
    "\n",
    "    #We checked for duplicates using the 'FictiveIDNumber' column, which uniquely identifies complaints. We noticed some complaints sharing the same 'FictiveIDNumber'.\n",
    "    #However, upon closer inspection, we found that these rows represent the same complainant reporting multiple charges in the StatisticGroup.\n",
    "    #To handle this, we can apply one-hot encoding to the StatisticGroup column, allowing each charge to be represented as a separate feature.\n",
    "\n",
    "    #Encoding the Data\n",
    "    categorical_columns = ['Yeshuv', 'PoliceStation']\n",
    "    data_encoded = pd.get_dummies(datasets, columns=categorical_columns, drop_first=True)\n",
    "\n",
    "    #Removing the duplicaite rows\n",
    "    data_encoded = data_encoded.drop_duplicates()\n",
    "\n",
    "    #Feature Engineering\n",
    "    data_encoded['Quarter_numeric'] = data_encoded['Quarter'].str.extract('(\\d)').astype(int)\n",
    "    data_encoded['Quarter_sin'] = np.sin(2 * np.pi * data_encoded['Quarter_numeric'] / 4)\n",
    "    data_encoded['Quarter_cos'] = np.cos(2 * np.pi * data_encoded['Quarter_numeric'] / 4)\n",
    "    data_encoded['YeshuvCrimeRate'] = datasets.groupby('Yeshuv')['Yeshuv'].transform('count')\n",
    "    data_encoded['CrimeTrend'] = data_encoded.groupby('Year')['Year'].transform('count')\n",
    "\n",
    "    # נרמול ערכים מספריים\n",
    "    numeric_columns = ['YeshuvCrimeRate', 'CrimeTrend']\n",
    "    if scaler is None:\n",
    "        scaler = MinMaxScaler()\n",
    "    if fit_scaler:\n",
    "        data_encoded[numeric_columns] = scaler.fit_transform(data_encoded[numeric_columns])\n",
    "    else:\n",
    "        data_encoded[numeric_columns] = scaler.transform(data_encoded[numeric_columns])\n",
    "\n",
    "    #Now we can remove the ID column\n",
    "    data_encoded = data_encoded.drop(columns=['FictiveIDNumber','Quarter'], errors='ignore')\n",
    "\n",
    "    return data_encoded, train_mappings, scaler"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "authorship_tag": "ABX9TyOxYpdUPNELkcvmFGN8W2A4",
   "include_colab_link": true,
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}