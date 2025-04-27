import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Sales Prediction Application')

st.sidebar.header('User Input:')

file = st.sidebar.file_uploader('Upload CSV File', type = ['csv'])

if file is not None:
    data = pd.read_csv(file)

    st.write('Uploaded File:', data.head())

    features = data[['TV', 'Radio', 'Newspaper']]
    target = data['Sales']

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

    model = LinearRegression()

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    st.write('Mean Absolute Error = ', mean_absolute_error(y_test, y_pred))
    st.write('R2 Score = ', r2_score(y_test, y_pred))

    fig,ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')

    #show plot
    st.pyplot(fig)


#st.write(data.describe())

st.sidebar.write('Please, Enter Your Data for PRediction.')

def user_input():
    TV = st.sidebar.slider('TV', 0.0, 300.0, 150.0)
    Radio = st.sidebar.slider('Radio', 0.0, 50.0, 25.0)
    Newspaper = st.sidebar.slider('Newspaper', 0.0, 120.0, 60.0)

    data = {
        'TV' : TV,
        'Radio' : Radio,
        'Newspaper' : Newspaper
    }

    features = pd.DataFrame([data])
    return features

input_df = user_input()

if st.sidebar.button('Predict Sales'):
    predcition = model.predict(input_df)
    st.sidebar.write('Sales Prediction:', predcition)


#txt = st.text_area('Enter a text')
