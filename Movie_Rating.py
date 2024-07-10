
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MultiLabelBinarizer
from sklearn import linear_model
import streamlit as st

import warnings
warnings.filterwarnings(category=UserWarning,action='ignore')

data=pd.read_excel('Movie_Rating.xlsx','movie_rating')
df=pd.DataFrame(data)

df=df.drop(['Genre','Poster_Link','Series_Title','Released_Year','Overview','Director','Star1','Star2','Star3','Star4'],axis='columns')
df.Gross=df.Gross.fillna(np.mean(df.Gross))
le=LabelEncoder()
mlb = MultiLabelBinarizer()

df['Runtime'] = df['Runtime'].astype(str).str.replace(' min', '').astype(int)
df['Certificate'] = le.fit_transform(df['Certificate'].astype(str))

x=df.drop(['IMDB_Rating'],axis='columns')
x=x.fillna(0)
y=df.IMDB_Rating
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.9)

le=linear_model.LinearRegression()
model=le.fit(xtrain,ytrain)

st.title('Movie Rating')
a=st.number_input('Certificate',min_value=0)
b=st.number_input('Runtime',min_value=0)
c=st.number_input('Meta_score',min_value=0)
d=st.number_input('No_of_Votes',min_value=0)
e=st.number_input('Gross',min_value=0)

if st.button('Predict'):
    df=pd.DataFrame({
        'Certificate':[a],
        'Runtime':[b],
        'Meta_score':[c],
        'No_of_Votes':[d],
        'Gross':[e]
    })
    st.write(model.predict(df))
				