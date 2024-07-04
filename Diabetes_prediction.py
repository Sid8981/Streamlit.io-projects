# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns



df = pd.read_csv("diabetes.csv")

# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())


# X AND Y DATA
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# Convert feature names to lowercase
df.columns = map(str.lower, df.columns)

# Train model with selected features
selected_features = ['age', 'bmi', 'bloodpressure', 'diabetespedigreefunction', 'glucose', 'insulin', 'pregnancies', 'skinthickness']
x_train, x_test, y_train, y_test = train_test_split(df[selected_features], df['outcome'], test_size=0.2, random_state=0)

# Function to get user data
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'age': age,
        'bmi': bmi,
        'bloodpressure': bp,
        'diabetespedigreefunction': dpf,
        'glucose': glucose,
        'insulin': insulin,
        'pregnancies': pregnancies,
        'skinthickness': skinthickness
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# Train model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Predict user data
user_result = rf.predict(user_data)

# Visualizations
st.title('Visualised Patient Report')

# Scatterplots for selected features only
for feature in selected_features:
    st.header(f'{feature.capitalize()} Value Graph (Others vs Yours)')
    fig = plt.figure()
    ax = sns.scatterplot(x='age', y=feature, data=df, hue='outcome', palette='rocket')
    ax = sns.scatterplot(x=user_data['age'], y=user_data[feature], s=150, color='red' if user_result[0] == 1 else 'blue')
    plt.xticks(np.arange(10, 100, 5))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig)

# OUTPUT
st.subheader('Your Report: ')
output = 'You are not Diabetic' if user_result[0] == 0 else 'You are Diabetic'
st.title(output)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test)) * 100) + '%')
