import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

#import datasets
train_df = pd.read_csv('archive/Train.csv')
test_df = pd.read_csv('archive/Test.csv')

#define the preprocessing function
def preprocess(df):
    #fill missing values
    df['Alcohol Consumption'] = df['Alcohol Consumption'].fillna('Never')
    df['Family History'] = df['Family History'].fillna('None')
    df['Chronic Diseases'] = df['Chronic Diseases'].fillna('None')
    df['Medication Use'] = df['Medication Use'].fillna('None')
    df['Education Level'] = df['Education Level'].fillna('None')
    
    #replace the blood pressure column with two separate columns
    df[['Systolic', 'Diastolic']] = df['Blood Pressure (s/d)'].str.split('/', expand=True)
    df['Systolic'] = pd.to_numeric(df['Systolic'])
    df['Diastolic'] = pd.to_numeric(df['Diastolic'])
    df = df.drop(columns=['Blood Pressure (s/d)'])
    
    return df

# apply preprocessing to datasets
train_df = preprocess(train_df)
test_df = preprocess(test_df)

#encode label columns
label_columns = ["Gender", "Physical Activity Level", "Smoking Status",
                 "Alcohol Consumption", "Diet", "Medication Use",
                 "Family History", "Chronic Diseases", "Mental Health Status",
                 "Sleep Patterns", "Education Level", "Income Level"]
le = LabelEncoder()
for column in label_columns:
    train_df[column] = le.fit_transform(train_df[column])
    test_df[column] = le.fit_transform(test_df[column])

#find the main correlators of age
train_df.corr()['Age (years)']
#visualization of correlators
plt.figure(figsize=(12, 8))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')

#LINEAR REGRESSION
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#split the train dataset
X_train = train_df.drop(columns=['Age (years)'])
y_train = train_df['Age (years)']
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#build the model
model = LinearRegression()
model.fit(X_train_split, y_train_split)
y_val_pred = model.predict(X_val_split)

#measure error
mse = mean_squared_error(y_val_split, y_val_pred)
variance = np.var(y_train)
r2 = r2_score(y_val_split, y_val_pred)

#make a prediction
y_pred = model.predict(test_df)

#import the results
test_predictions = pd.DataFrame({'Predicted Age': y_pred})
test_predictions.to_csv('lr-predictions.csv', index=False)

#RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor

#build the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_split, y_train_split)
y_val_pred_rf = rf_model.predict(X_val_split)

#measure error
mse_rf = mean_squared_error(y_val_split, y_val_pred_rf)
r2_rf = r2_score(y_val_split, y_val_pred_rf)

#make a prediction
y_pred_rf = rf_model.predict(test_df)

#import the results
test_predictions_rf = pd.DataFrame({'Predicted Age': y_pred_rf})
test_predictions_rf.to_csv('rf-predictions.csv', index=False)
