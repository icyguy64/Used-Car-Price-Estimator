#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import pickle

st.write("""
# Predicting Used Car Price Estimator deployed using streamlit
For this project, I made use of Craglist's Used Car Price dataset to build a linear regression model to determine price of a used car based on its features. The objective of this project is to experiment with various regression methods to determine which model yields the highest accuracy and also determine the features which are most indicative of a used car being highly priced.
""")

st.sidebar.header('Car Features')

# Functionalize model fittting
import pickle

def user_input_features():
    V1 = st.sidebar.slider('Year', 111, 121, 114)
    V2 = st.sidebar.slider('Manufacturer', 0, 41, 19)
    V3 = st.sidebar.slider('Condition', 0.0, 5.0, 1.0)
    V4 = st.sidebar.slider('Cylinders', 0.0, 7.0, 4.4)
    V5 = st.sidebar.slider('Fuel', 0.0, 4.0, 2.0)
    V6 = st.sidebar.slider('Odometer', 0.0, 1629.0, 16.0)
    V7 = st.sidebar.slider('Transmission', 0.0, 2.0, 0.1)
    V8 = st.sidebar.slider('Drive', 0.0, 2.0, 0.7)
    V9 = st.sidebar.slider('Type', 0.0, 12.0, 6.1)
    V10 = st.sidebar.slider('Paint Color', 0.0, 11.0, 5.7)
    V11 = st.sidebar.slider('State', 0.0, 50.0, 24.1)
    data = {'Year': V1,
           'Manufacturer': V2,
           'Condition': V3,
           'Cylinder': V4,
           'Fuel': V5,
           'Odometer': V6,
           'Transmission': V7,
           'Drive': V8,
           'Type': V9,
           'Paint Color': V10,
           'State': V11
           }

    features= pd.DataFrame(data, index=[0])
    return features
df =  user_input_features()
st.subheader('Car Features')
st.write(df)

filename = 'linear_svr'
model = pickle.load(open(filename,'rb'))
preprocess = pickle.load(open('preprocess_scale','rb'))
pred = model.predict(preprocess.transform(df))
st.subheader('Prediction - estimated price of the used car')
st.write(pred)
