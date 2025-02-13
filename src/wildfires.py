import streamlit as st
from pickle import load
import pandas as pd
import numpy as np
import pydeck as pdk
from datetime import date
from sklearn.preprocessing import RobustScaler



@st.cache_resource
def load_model():
    return load(open('../models/modelo_xgb_ignition.sav', 'rb'))

modelo = load_model()

@st.cache_data
def load_data():
    return pd.read_csv('../data/processed/incendio_cleaned.csv', usecols=['latitud', 'longitud', 'altitud', 'tempmaxima', 'humrelativa', 'diasultimalluvia'])

df = load_data()

scaler = RobustScaler() 

scaler.fit(df[['altitud', 'tempmaxima', 'humrelativa', 'diasultimalluvia']])


col1, col2 = st.columns([4, 8])    

with col1:
    st.image("fireriskai.png", width=100)

with col2:
    st.title('FireRisk AI')

st.write('Sistema inteligente que analiza datos ambientales en tiempo real para predecir el riesgo de ignición de un incendio forestal')


#Mapa de incendios

if "mapa" not in st.session_state:
    #with st.expander("🗺️ Ver mapa de incendios", expanded=True):

    st.subheader('🗺️ Mapa de incendios')
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["longitud", "latitud"],      # Asegúrate del orden: [longitud, latitud]
        get_radius=500,                   # Radio en metros (puedes ajustarlo)
        get_fill_color=[255, 0, 0, 160],    # Color rojo (RGBA)
        pickable=True
    )

    #Centrar el mapa en el centro de España
    view_state = pdk.ViewState(
        latitude=40,
        longitude=-3,
        zoom=5,
        pitch=0
    )

    # Crear el mapa con pydeck
    deck = pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=view_state,
        layers=[layer],
        tooltip={"text": "Incendio"}
    )

st.pydeck_chart(deck)
st.text('Ubicación de los incendios registrados en España entre 1968 y 2016')

with st.form("datos usuario"):
    st.subheader('📍 Ubicación')

    provincias = [
    "Álava", "Albacete", "Alicante", "Almería", "Asturias", "Ávila", "Badajoz", "Barcelona",
    "Burgos", "Cáceres", "Cádiz", "Cantabria", "Castellón", "Ciudad Real", "Córdoba", "Cuenca",
    "Gerona", "Granada", "Guadalajara", "Guipúzcoa", "Huelva", "Huesca", "Islas Baleares",
    "Jaén", "La Coruña", "La Rioja", "Las Palmas", "León", "Lérida", "Lugo", "Madrid", "Málaga",
    "Murcia", "Navarra", "Orense", "Palencia", "Pontevedra", "Salamanca", "Santa Cruz de Tenerife",
    "Segovia", "Sevilla", "Soria", "Tarragona", "Teruel", "Toledo", "Valencia", "Valladolid",
    "Vizcaya", "Zamora", "Zaragoza"
    ]
    provincia = st.selectbox('Provincia', provincias)
    latitud = st.slider('Latitud', min_value=27.63, max_value=43.79, value=30.00, step=0.10)
    longitud = st.slider('Longitud', min_value=-18.17, max_value=4.36, value=-10.00, step=0.10)
    altitud = st.slider('Altitud (m)', min_value=0, max_value=3715, value=1000, step=10)

    st.subheader('🌤 Condiciones meteorológicas')
    tempmaxima = st.slider('Temperatura máxima (ºC)', min_value=-10, max_value=60, value=20, step=1)
    humrelativa = st.slider('Humedad relativa (%)', min_value=0, max_value=100, value=50, step=1)
    ultimalluvia = st.slider('Días desde la última lluvia', min_value=0, max_value=365, value=30, step=1)

    st.subheader('🕒 Hora y mes del posible foco')
    col3, col4 = st.columns(2)

    hora = ["Mañana", "Tarde", "Noche", "Madrugada"]
    mes = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]

    with col3:
        horadeteccion = st.selectbox('Hora', hora)

    with col4:
        mesdeteccion = st.selectbox('Mes', mes)

    submitted = st.form_submit_button("🔥 Calcular riesgo de incendio")


# Preprocesamiento de los datos

if submitted:

    scaled_values = scaler.transform(np.array([[altitud, tempmaxima, humrelativa, ultimalluvia]]))
    altitud_r, tempmaxima_r, humrelativa_r, diasultimalluvia_r = scaled_values[0]


    deteccion_n = 1827.0

    provincia_n = pd.factorize([provincia])[0][0]
    horadeteccion_n = pd.factorize([horadeteccion])[0][0]
    mesdeteccion_n = pd.factorize([mesdeteccion])[0][0]


    user_input = np.array([[latitud, longitud, altitud_r, diasultimalluvia_r, deteccion_n, tempmaxima_r, humrelativa_r, provincia_n, mesdeteccion_n, horadeteccion_n]])

    prob = modelo.predict(user_input)[0]

    if prob < 0:
        st.error("🚫 Introduzca otra ubicación.")
    elif 0 <= prob <= 19:
        st.success(f'El riesgo de un incendio forestal es del {prob:.2f}% - **Riesgo Bajo** 🟩')
    elif 20 <= prob <= 39:
        st.warning(f'El riesgo de un incendio forestal es del {prob:.2f}% - **Riesgo Medio** 🟩🟨')
    elif 40 <= prob <= 59:
        st.warning(f'El riesgo de un incendio forestal es del {prob:.2f}% - **Riesgo Alto** 🟩🟨🟧')
    else:
        st.error(f'El riesgo de un incendio forestal es del {prob:.2f}% - **Riesgo Extremo** 🟩🟨🟧🟥')
    