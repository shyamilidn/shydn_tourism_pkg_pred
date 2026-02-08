# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("shydnTPkg"))
DATASET_PATH = "hf://datasets/TPP/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

df.rename(columns={'Unnamed: 0': 'Col1'}, inplace=True)

# Drop unique identifier column (not useful for modeling)
df.drop(columns=['Col1', 'CustomerID','Age', 'TypeofContact', 'Occupation','Gender'
                  'NumberOfPersonVisiting', 'PreferredPropertyStar', 'MaritalStatus',
                 'NumberOfTrips', 'Passport', 'NumberOfChildrenVisiting',
                 'Designation'], inplace=True)

df['ProdTaken'] = df['ProdTaken'].astype('category')
df['CityTier'] = df['CityTier'].astype('category')
df['OwnCar'] = df['OwnCar'].astype('category')

# Encode categorical columns
label_encoder = LabelEncoder()
df['ProdTaken'] = label_encoder.fit_transform(df['ProdTaken'])
df['CityTier'] = label_encoder.fit_transform(df['CityTier'])
df['ProductPitched'] = label_encoder.fit_transform(df['ProductPitched'])
df['OwnCar'] = label_encoder.fit_transform(df['OwnCar'])

# Define target variable
target_col = 'PitchSatisfactionScore'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.275, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="shyam/TPP",
        repo_type="dataset",
    )
