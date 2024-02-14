import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

class_names = ['Pepper__bell___Bacterial_spot',
 'Pepper__bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

# Load your trained model
def load_model():
    model = tf.keras.models.load_model('model_version1')
    return model

# Function to make predictions
def predict(model, img):
    # Preprocess the image
    img_array = tf.image.decode_image(img, channels=3)
    img_array = tf.image.resize(img_array, (256,256))
    img_array = tf.expand_dims(img_array, 0)  # Create a batch dimension

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    return predicted_class, confidence

def main():
    st.title('Tomatiki Plant Disease Detection (Tomato , Potato , Pepper bell)')
    
    # Load the model
    model = load_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image as bytes
        img_bytes = uploaded_file.read()
        st.image(img_bytes, caption='Uploaded Image', use_column_width=True)

        # Button for prediction
        if st.button('Predict'):
            # Make predictions
            predicted_class, confidence = predict(model, img_bytes)

            st.write("Status:", class_names[predicted_class].replace("_", " "))
            st.write("Confidence:", confidence*100 , " %")

if __name__ == "__main__":
    main()
