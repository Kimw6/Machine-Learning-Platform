import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from PIL import Image


def load_model():
    model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
    return model

model = load_model()

st.subheader("`Image Classification with ResNet50`")
st.write("This is pre-trained on the ImageNet dataset. More information about the dataset can be found [here](http://www.image-net.org/).")
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
    return decode_predictions(predictions, top=5)

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)  
    processed_image = process_image(image)

    try:
        with st.spinner("Predicting..."):
            label = make_prediction(processed_image, model)

    except Exception as e:
        st.error(f"Error: {e}")
    else:
        st.write("***Top 5 predictions:***")
        for i in label[0]:
            st.write(f"{i[1]}: {i[2]*100:.2f}%")

