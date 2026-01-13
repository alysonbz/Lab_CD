import pandas as pd

# Load the metadata file
df = pd.read_csv('Chest_xray_Corona_Metadata.csv')

# Display the first few rows and summary of Dataset_type
print(df.head())
print(df['Dataset_type'].value_counts())
print(df['Label'].value_counts())