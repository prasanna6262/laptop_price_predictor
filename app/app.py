import streamlit as st
import joblib 
import numpy as np
model=joblib.load("../models/laptop_price_model.pkl")
st.title('Laptop Price Pridictor')
brand=st.selectbox("Brand",['Dell','Hp','Lenovo','Asus'])
model_name=st.selectbox("Model Name", ['Inspiron15','Think pad','Pavilion','ViviBook'])
processor=st.selectbox("Processor",['Intel i5','intel i7','ADM Ryzen 5','Apple M1'])
os=st.selectbox("OS",['Windows 11','Windows 11','Windows 10','Windows 10'])
ram=st.slider("RAM(GB)",8,64,step=4)
storage=st.slider("SSd Storage(GB)",128,2000,step=128)
screen_size=st.slider("Screen Size(inches)",11.0,17.0,step=0.1)
gpu=st.selectbox("GPU",['Intel UHD','NVIDIA','AMD'])
weight=st.number_input("Weight (kg)",min_value=1.0,max_value=5.0,step=0.1)
# Dummy encoding for demo
brand_encoded=['Dell','Hp','Lenovo','Asus'].index(brand)
processor_encoded=['Intel i5','intel i7','ADM Ryzen 5','Apple M1'].index(processor)
os_encoded=['Windows 11','Windows 11','Windows 10','Windows 10'].index(os)
gpu_encoded=['Intel UHD','NVIDIA','AMD'].index(gpu)
model_name_encoded=['Inspiron15','Think pad','Pavilion','ViviBook'].index(model_name)
features=np.array([[brand_encoded,processor_encoded,ram,storage,screen_size,gpu_encoded,os_encoded,weight,model_name_encoded]])
st.write("Input features:",features)
st.write("Shape:",features.shape)
price=model.predict(features)[0]
st.success(f"Estimated Price: ₹{int(price)}")