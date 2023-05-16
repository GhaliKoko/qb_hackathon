import streamlit as st
import numpy as np
import rasterio
from tensorflow.keras.models import load_model
import pandas as pd
from PIL import Image

# Set page config
st.set_page_config(page_title='Methane Plume Detection', layout='wide', initial_sidebar_state='auto')

# Load the images
image1 = Image.open('Table.png')
image2 = Image.open('column1.png')
image3 = Image.open('column2.png')
image4 = Image.open('Map.png')
image5 = Image.open('Historical data.png')



# Function to load model
@st.cache_resource
def load_keras_model(path):
    model = load_model(path, compile=False)
    return model


# Function to read .tif file
def read_tif(file_path):
    with rasterio.open(file_path) as src:
        return src.read(1)


# Load the model
model_path = 'my_model.h5'
qb_model = load_keras_model(model_path)

# Define pages
PAGES = {
    "Home Menu": "home_menu",
    "Automate Detection": "automate_detection",
    "Leakage Detection": "leakage_detection",
    "Historical Data": "historical_data"
}

# Sidebar - Page Selection
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# Page title
st.title(selection)

if selection == "Home Menu":
    st.markdown("# Global Methane Tracker and Automatic Detection")
    st.subheader('Overview')
    # Display the images
    st.image(image1, use_column_width=True)

    # Display the images side by side in columns
    col1, col2 = st.columns(2)

    with col1:
        st.image(image2,use_column_width=True)

    with col2:
        st.image(image3, use_column_width=True)

    st.image(image4, use_column_width=True)

elif selection == "Automate Detection":
    # Main
    # Upload image
    uploaded_file = st.file_uploader("Choose an image file", type=['tif'])

    if uploaded_file is not None:
        # Open the image
        image = read_tif(uploaded_file)

        # Normalize the image
        image = image / 255.0

        # Reshape the image to have a channel dimension and match the model input shape
        image = np.expand_dims(image, axis=[0, -1])

        # Predict
        prediction = qb_model.predict(image)

        # Show the image
        st.subheader('Uploaded Image')
        st.image(image[0, ..., 0], caption='Uploaded Image.', width=300, clamp=True)

        # Show the prediction
        st.subheader('Prediction')
        if prediction < 0.5:
            st.success("This image does NOT contain a methane plume.")
        else:
            st.error("This image contains a methane plume.")

elif selection == "Leakage Detection":
    longitude = st.sidebar.number_input('Enter longitude:', value=0.0)
    latitude = st.sidebar.number_input('Enter latitude:', value=0.0)
    # Libya
    df_libya = pd.DataFrame(
        np.random.randn(250, 2) / [70, 70] + [27.0, 17.0],
        columns=['lat', 'lon'])

    # Middle East (around Saudi Arabia)
    df_mid_east = pd.DataFrame(
        np.random.randn(250, 2) / [70, 70] + [24.0, 45.0],
        columns=['lat', 'lon'])

    # Russia
    df_russia = pd.DataFrame(
        np.random.randn(250, 2) / [70, 70] + [61.5240, 105.3188],
        columns=['lat', 'lon'])

    # Indonesia
    df_indonesia = pd.DataFrame(
        np.random.randn(250, 2) / [70, 70] + [-0.7893, 113.9213],
        columns=['lat', 'lon'])

    df_usa = pd.DataFrame(
        np.random.randn(1000, 2) / [70, 70] + [37.76, -122.4],
        columns=['lat', 'lon'])

    # Concatenate all the dataframes
    df = pd.concat([df_libya, df_mid_east, df_russia, df_indonesia, df_usa])
    st.map(df, zoom=1)


else:

    # include the code for the historical data page here
    historical_data_options = ["Past 30 days", "Past 70 days", "Past 1 year"]
    option = st.selectbox("Select option:", historical_data_options)
    st.image(image5, use_column_width=True)

