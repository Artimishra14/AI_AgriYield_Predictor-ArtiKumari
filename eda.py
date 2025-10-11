import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('projectdata_cleaned.csv')

print(df.describe())


numeric_features = ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall', 'yield']

for feature in numeric_features:
    plt.figure(figsize=(6,3))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()


plt.figure(figsize=(10,8))
corr_matrix = df[numeric_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.show()

sns.pairplot(df, hue='crop_std', vars=['n', 'p', 'k', 'temperature', 'yield'])
plt.show()
