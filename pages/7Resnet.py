import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from PIL import Image

# Load the pre-trained ResNet50 model
@st.cache_data()
def load_model():
    model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
    return model

model = load_model()

st.title("ResNet50 Image Classification")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def process_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    image = preprocess_input(tf.keras.preprocessing.image.img_to_array(image))
    image = tf.expand_dims(image, axis=0)
    return image

def make_prediction(image, model):
    predictions = model.predict(image)
    return decode_predictions(predictions, top=1)[0][0]

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    processed_image = process_image(image)

    # Make predictions
    try:
        with st.spinner("Predicting..."):
            label = make_prediction(processed_image, model)
    except Exception as e:
        st.error(f"Error: {e}")
    else:
        st.write(f"***Predicted class: `{label[1]}`***")
        st.write(f"***Confidence: {label[2]:.2%}***")
else:
    st.write("Please upload an image.")
