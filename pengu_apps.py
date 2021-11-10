import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import plotly.express as px

st.write("""
# Pengu Prediction App

This app predicts the **Palmer Penguin** species!

Data obtained from the [palmerpenguins dataset](https://www.kaggle.com/parulpandey/palmer-archipelago-antarctica-penguin-data).
""")

# Image 
st.image("https://github.com/allisonhorst/palmerpenguins/raw/master/man/figures/lter_penguins.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

st.header('Input Pemguin info')

def user_input_features():
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    sex = st.selectbox('Sex', ('male', 'female'))
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 45.85)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.3)
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.5)
    body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4500.0)
    data = {'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex}
    features = pd.DataFrame(data, index = [0])
    return features
input_df = user_input_features()

peng1 = pd.read_csv('./penguins_size.csv')
pengu = peng1.drop(columns = ['species'])
df = pd.concat([input_df, pengu], axis = 0)

encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix = col)
    df = pd.concat([df, dummy], axis = 1)
    del df[col]
df = df[:1] 

st.subheader('User Input features')
st.write(df)

clf = pickle.load(open('penguins_model.pkl', 'rb'))

predict = clf.predict(df)
predict_probability = clf.predict_proba(df)

st.subheader('Species Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[predict])

st.subheader('Prediction Probability  1:Adelie, 2:Chinstrap, 3:Gentoo')
st.write(predict_probability)

peng2 = pd.read_csv("./penguins_size.csv")

st.subheader('DATA_Example')
numshow = st.slider("Example: penguin table", 0, 350, 5)
st.table(peng2[:numshow])

st.subheader('EDA')
p = sns.pairplot(peng2, hue = 'species')
st.pyplot(p)

st.subheader("Mean body mass index distribution")
st.write(peng2.groupby(['species','sex']).mean()['body_mass_g'].round(2))


