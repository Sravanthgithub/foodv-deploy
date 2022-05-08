import streamlit as st

import tensorflow as tf
from PIL import Image
import img_classification
import numpy as np
st.set_page_config(page_title="Food Vision",
                   page_icon="🍔")

st.title("Food Vision 🍔📷")
st.header("Identify what's in your food photos!")

st.sidebar.title("What actually is this?")
st.sidebar.write("""
FoodVision is an end-to-end **CNN Image Classification Model** which identifies the food in your image. 
It can identify over 100 different food classes

And also this model is trained using Transfer Learning (Efficientnet-B0)
""")
st.sidebar.markdown("Created by **Sravanth**")
uploaded_file = st.file_uploader("Upload a food image", type=["jpeg","jpg","png"])
if uploaded_file is not None:
    img = uploaded_file.read()
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    #img = tf.io.read_file(uploaded_file)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    st.write("Classifying...")
    label = img_classification.classify(img)
    label = label.capitalize()
    st.success(f'Prediction : {label}\n')


