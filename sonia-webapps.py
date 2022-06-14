import streamlit as st
import pandas as pd
from PIL import Image
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB



st.write("""
# Web Apps - Sonia
Aplikasi Web Sonia
"""
)

img = Image.open('menhara.jpg')
st.image(img, use_column_width=False)

st.sidebar.header('Parameter Inputan')

def input_user():
    panjang_sepal = st.sidebar.slider('Panjang Sepal', 4.3, 7.9, 5.4)
    lebar_sepal   = st.sidebar.slider('Lebar Sepal', 2.0, 4.4, 3.4)
    panjang_petal = st.sidebar.slider('Panjang Petal', 1.0, 6.9, 5.4)
    lebar_petal   = st.sidebar.slider('Lebar Petal', 0.1, 2.5, 1.0)

    data = {'panjang sepal' : panjang_sepal,
            'lebar petal'   : lebar_sepal,
            'panjang_petal' : panjang_petal,
            'lebar-Petal'   : lebar_petal
    }

    fitur = pd.DataFrame(data, index=[0])
    return fitur

df = input_user()

st.subheader('Parameter Inputan')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

model = GaussianNB()
model.fit(X,Y)
 
prediksi = model.predict(df)
prediksi_proba = model.predict_proba(df)

st.subheader('Label kelas dan nomor indeks yang sesuai')
st.write(iris.target_names)

st.subheader('Prediksi (Hasil Spesifikasi)')
st.write(iris.target_names[prediksi])

st.subheader('Probabilitas Hasil Prediksi (Klasifikasi)')
st.write(prediksi_proba)