import streamlit as st

# Set page config
st.set_page_config(page_title='Methane Plume Detection', layout='wide', initial_sidebar_state='auto')

# Define pages
PAGES = {
    "Overview": "overview",
    "Automate Detection": "automate_detection",
    "Leakage Detection": "leakage_detection",
    "Historical Data": "historical_data"
}

# Sidebar - Page Selection
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# Page title
st.title(selection)

if selection == "Overview":
    st.subheader('This is the Overview page')


elif selection == "Automate Detection":
    st.subheader('This is the Automate Detection page')


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

    # Page title
    st.title('Methane Plume Detection')

    # Sidebar
    st.sidebar.header('Upload Image')


    # Main
    def main():
        # Upload image
        uploaded_file = st.sidebar.file_uploader("Choose an image file", type=['tif'])

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
                st.success("The model predicts this image does NOT contain a methane plume.")
            else:
                st.error("The model predicts this image contains a methane plume.")


    # Run the streamlit app
    if __name__ == "__main__":
        main()

elif selection == "Leakage Detection":
    st.subheader('This is the Leakage Detection page')
    # include the code for the leakage detection page here

else:
    st.subheader('This is the Historical Data page')
    # include the code for the historical data page here
    historical_data_options = ["Option 1", "Option 2", "Option 3"]
    option = st.selectbox("Select option:", historical_data_options)
    st.write("You selected:", option)
