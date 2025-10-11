import pandas as pd
df_faostat = pd.read_csv('FAOSTAT_data_en_10-5-2025.csv')
df_kaggle = pd.read_csv('Crop_recommendation.csv')
df_foodgrains = pd.read_csv('Foodgrains1.csv')
df_oilseeds = pd.read_csv('oilseeds1.csv')

df_foodgrains.columns = df_foodgrains.columns.str.strip()
df_oilseeds.columns = df_oilseeds.columns.str.strip()

def map_crop_names(name):
    crop_name_map = {
        'Maize (corn)': 'maize',
        'Rice': 'rice',
        'Wheat': 'wheat',
        'Gram': 'chickpea',
        'Mung (Green Gram)': 'mungbean',
        'Udad': 'blackgram',
        'Tur (Red Gram)': 'pigeonpeas',
        'Jowar': 'sorghum',
        'Bajra': 'pearlmillet',
        'Ragi': 'fingermillet',
        'Math ': 'mothbeans',
        'Other Pulses': 'other pulses',
        'Soyabean': 'soybean',
        'Groundnut': 'groundnut',
        'Castor seed': 'castor',
        'Sesamum': 'sesame',
        'Rapeseed & Mustard': 'mustard',
    }
    return crop_name_map.get(name.strip(), name.strip().lower().replace(' ', '').replace('&', 'and'))

df_faostat['crop_std'] = df_faostat['Crop'].apply(map_crop_names)
df_foodgrains['crop_std'] = df_foodgrains['Crop'].apply(lambda x: x.strip().lower())
df_oilseeds['crop_std'] = df_oilseeds['Crops'].apply(lambda x: x.strip().lower())
df_kaggle['crop_std'] = df_kaggle['label'].apply(lambda x: x.strip().lower())


kaggle_filtered = df_kaggle[df_kaggle['crop_std'].isin(df_faostat['crop_std'])]


merged_df = pd.merge(kaggle_filtered, df_faostat, on='crop_std', how='inner')


df_foodgrains = df_foodgrains.rename(columns={'Yield': 'Yield_foodgrains'})
df_oilseeds = df_oilseeds.rename(columns={'Yield': 'Yield_oilseeds'})


merged_df = pd.merge(merged_df, df_foodgrains[['crop_std', 'Yield_foodgrains']], on='crop_std', how='left')
merged_df = pd.merge(merged_df, df_oilseeds[['crop_std', 'Yield_oilseeds']], on='crop_std', how='left')


merged_df.to_csv('final_agri_full_dataset.csv', index=False)

print('Merged dataset saved to final_agri_full_dataset.csv')
