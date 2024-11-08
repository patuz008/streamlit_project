import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly.express as px
warnings.filterwarnings('ignore')

# Scikit-learn libraries for preprocessing, models, and evaluation
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score, accuracy_score,precision_score, confusion_matrix,RocCurveDisplay,\
ConfusionMatrixDisplay, roc_auc_score, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_selection import RFE

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

pd.set_option('display.max_columns', None)

df = pd.read_csv('kidney_disease.csv')

header = st.container()
dataset = st.container()
analysis = st.container()
predictive_model = st.container()
model_training = st.container()

with header:
    st.title ("This is a Data collection")
    st.text("PROJECT")
with dataset:
    st.title ("Health data")
    st.text("Analyse dataset")
    data_tab = pd.read_csv("kidney_disease.csv")
    st.write(data_tab.head())

with analysis:
    st.subheader('Patient Classification')
    data_tab1 = data_tab.dropna(inplace=True)
    # data_tab1.sort_values(by = 'age', ascending = False)
    st.write(data_tab.head())
    patient_classification = pd.DataFrame(data_tab['classification'].value_counts())
    st.text('Classification Distribution between notckd and ckd')
    st.bar_chart(patient_classification)
    class_distribution_percentage = pd.DataFrame(data_tab['classification'].value_counts())
    
    # fig, ax = plt.subplots()
    # ax.pie(class_distribution_percentage)
    # st.pyplot(patient_classification)
    # st.write ('Pie Chart')
    # fig = px.pie(data_tab1, values=class_distribution_percentage)
    # st.plotly_chart(fig, use_container_width=True)
    # class_distribution_percentage.plot(kind='pie', autopct='%1.2f%%', subplots=False)
    
    # Get the value counts of 'age' and store it in a variable (e.g., age_counts)
    age_counts = pd.DataFrame(data_tab['age'].value_counts().sort_index(ascending=False))
    st.text('Relationship between age and bp')
    # st.write(age_counts.head(20).plot(kind='bar', title='Relationship between age and bp'))
    st.bar_chart(age_counts)  

    hemo_levels = pd.DataFrame(data_tab['hemo'].value_counts())
    
    st.scatter_chart(hemo_levels)


with predictive_model:
    st.title ("Predictive Modeling")
    st.text("we are creating a predictive model for our dataset")
    def clean_column(column):
        if column.dtype == "object":  # Check if column is non-numeric
            return column.str.replace(r'\s+', '', regex=True)  # Remove whitespace
        return column
    data_tab2 = df.apply(clean_column)

    numerical_columns = ['id', 'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv']
    df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')
    
    missing_values_count = data_tab2.isnull().sum()

    # Calculate the total number of rows
    total_rows = len(data_tab2)

    # Calculate the percentage of missing values for each column
    missing_values = ((missing_values_count / total_rows) * 100).sort_values(ascending=False)

    # missing_values

    categorical_columns = []
    numerical_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            categorical_columns.append(col)
        else:
            numerical_columns.append(col)

    data_tab2.drop('id', axis=1, inplace=True)
    train, test = train_test_split(df, test_size=0.3, random_state=42)
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    target = 'classification'

    X_train = train.drop(target, axis=1)
    y_train = train[target]

    X_test = test.drop(target, axis=1)
    y_test = test[target]


    cat = []
    num = []

    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            cat.append(col)
        else:
            num.append(col)
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    encoded_data = enc.fit_transform(X_train[cat])
    encoded_df = pd.DataFrame(encoded_data, columns=cat)
    X_train = pd.concat([X_train[num], encoded_df], axis=1)

    encoded_data2 = enc.transform(X_test[cat])
    encoded_df2 = pd.DataFrame(encoded_data2, columns=cat)
    X_test = pd.concat([X_test[num], encoded_df2], axis=1)


    # DT_model = model_to_use("DT")
    # DT_model.fit(X_train, y_train)
    # performance(DT_model,X_train,y_train,X_test, y_test)

    st.text('Our cleaned data')
    st.write(data_tab2.head(6))
    st.subheader('Our trained data')
    st.write(train.head(5))
    st.text('Whether Patient is:')
    hypertensive = st.selectbox('Hypertensive or Not?', df['htn'].unique(), index=0)

    Patient_diagnosis = st.selectbox('Is the diagnisis chronic or NOT?', df['classification'].unique(), index=1)



