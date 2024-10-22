

import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image
import pickle
from keras.layers import GlobalAveragePooling2D
from keras.models import Model

image_size = (224, 224)

image_name = "user_eye.png"

PREDICTION_LABELS = ["Cataract", "Normal"]

PREDICTION_LABELS.sort()

CSS_STYLE = """
    <style>
    .stApp {
        background-color: #a18a89; /* Change this to any color you prefer */
    }
    [data-testid="stSidebar"]{
        background-color: #e0ded3;  /* Change this to any color you prefer */
    }
    header, .css-1v3fvcr, .css-fg4pbf {
        background-color: #a18a89; /* Ensures top bar matches background */
    }
    </style>
    """
# set css styles
st.markdown(CSS_STYLE, unsafe_allow_html=True)


#write functions and come back when we need to write them
@st.cache_resource
#HW: read up on on caching
def get_convext_model():
  base_model = tf.keras.applications.ConvNeXtXLarge(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  model_frozen = Model(inputs=base_model.input,outputs=x)
  return model_frozen

@st.cache_resource
def load_sklearn_models(model_path):
    with open(model_path, 'rb') as model_file:
        final_model = pickle.load(model_file)
    return final_model


def featurization(image_path, model):
  img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  img_batch = np.expand_dims(img_array, axis=0)
  img_preprocessed = preprocess_input(img_batch)
  predictions = model.predict(img_preprocessed)
  return predictions


convext_featurized_model = get_convext_model()
cataract_model = load_sklearn_models("ConvNexXtlarge_MLP_best_model (1)")

# Beginning of styling portions

st.sidebar.title(":rainbow[**WELCOME**]")

page = st.sidebar.radio("Go to", ["**Home**", "**About Us**", "**How App Works**", "**Future Enhancement**","**Our Impact**"])

if page == "**Home**":
    st.markdown("<h1 style='color: white;'>Welcome to Our Cataract Image Predictor App</h1>", unsafe_allow_html=True) #or use this

    # Home page content
    st.image(
        "https://mediniz-images-2018-100.s3.ap-south-1.amazonaws.com/post-images/chokhm_1663869443.png",
        caption="Cataract Prediction"
    )

    st.header("About the web app")
    st.write("The Web App helps predict, from the image, whether or not the user has cataract.")

    tab1, tab2 = st.tabs(["Image Upload 👁️", "Camera Upload 📷 (coming soon in future)"])
    with tab1:
        image = st.file_uploader(label="Upload an image", accept_multiple_files=False, help="Upload an image to classify them")
        if image:
            image_type = image.type.split("/")[-1]
            if image_type not in ['jpg', 'jpeg', 'png', 'jfif']:
                st.error(f"Invalid file type : {image.type}", icon="🚨")
            else:
                user_image = Image.open(image)
                user_image.save("user_eye.png")
                st.image(user_image, caption="Uploaded Image")
                with st.spinner("Processing..."):
                    image_features = featurization("user_eye.png", convext_featurized_model)
                    model_predict = cataract_model.predict(image_features)
                    model_predict_proba = cataract_model.predict_proba(image_features)
                    probability = model_predict_proba[0][model_predict[0]]
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Disease Type")
                    st.subheader(PREDICTION_LABELS[model_predict[0]])
                with col2:
                    st.header("Prediction Probability")
                    st.subheader(f"{probability:.2f}")

    with tab2:
        cam_image = st.camera_input("Take a photo of the eye")
        if cam_image:
            user_image = Image.open(cam_image)
            user_image.save("user_eye.png")
            st.image(user_image, caption="Captured Image")
            with st.spinner("Processing..."):
                image_features = featurization("user_eye.png", convext_featurized_model)
                model_predict = cataract_model.predict(image_features)
                model_predict_proba = cataract_model.predict_proba(image_features)
                probability = model_predict_proba[0][model_predict[0]]
            col1, col2 = st.columns(2)
            with col1:
                st.header("Disease Type")
                st.subheader(PREDICTION_LABELS[model_predict[0]])
            with col2:
                st.header("Prediction Probability")
                st.subheader(f"{probability:.2f}")

elif page == "**About Us**":
    st.title("*About Us* 😊") #you can use keyboard shortcuts to get emojis
    st.markdown("<h2 style='color: blue;'>🌐 Who We Are: YouthForElders</h2>", unsafe_allow_html=True)

    st.write("""
        We are a dedicated team aiming to help underserved communities by providing access to vital medical care.
        Our mission is to use AI technology to bring healthcare solutions to areas with limited resources.
    """)

elif page == "**How App Works**":

  st.title("*How the App Works*🔔") #you can use keyboard shortcuts to get emojis

  st.write("""
        This web app utilizes deep learning models like ConvNeXt for feature extraction from eye images,
        followed by classification using a machine learning model. The app processes the image, extracts
        relevant features, and predicts whether the image shows signs of cataract.

        **Steps**:
        1. Upload or capture an image of the eye.
        2. The image is processed using a pre-trained deep learning model.
        3. The machine learning model classifies the image as either 'Cataract' or 'Normal'.
        4. The result is displayed along with a probability score.
    """)

elif page == "**Future Enhancement**":
  st.title("*Future Enhancements* 💡")
  st.write("""In the future, we aim to:
        - Improve the model's accuracy by including more diverse datasets.
        - Add more eye disease classifications like glaucoma and diabetic retinopathy.
        - Allow real-time processing and enhanced feedback for users.
        - Provide options for users to track their eye health over time.""")


elif page == "**Our Impact**":
  st.title("*Our Impact* ✅ ")
  st.write("""Global Reach: Briefly talk about how you aim to help underserved communities globally, using examples like India (one doctor per 10,000 people).
    Data-Driven: Highlight any data or impact metrics you’ve collected or plan to.""")

  # st.title("Cataract Image Predictor")

  # st.image(
  #     "https://mediniz-images-2018-100.s3.ap-south-1.amazonaws.com/post-images/chokhm_1663869443.png",
  #     caption = "Cataract Eyes")

  # st.header("About the web app")
  # with st.expander("Web App 🌐"):
  #   st.subheader("Eye disease classification")
  #   st.write("The Web App helps predict, from the image, whether or not the user has cataract")

  # with st.expander("How to use"):
  #   st.write("1. Scroll down to the 2 tabs that allow you to upload an image or take a picture if your eye")
  #   st.write("2. Click on the method that you want to use")
  #   st.write("3. For uploading an image, click on the 'Browse files' button that is present and choose the fundus image you want to upload")
  #   st.write("4. For taking a picture, position your eye so that the eye is within the camera frame and click 'Take photo'")
  #   st.write("5. Wait for the model to detect the cataracts, and scroll down for the diagnosis")

  # tab1, tab2 = st.tabs(["Image Upload 👁️", "Camera Upload 📷"])
  # with tab1:
  #   image = st.file_uploader(label="Upload an image",accept_multiple_files=False, help="Upload an image to classify them")
  #   if image:
  #     image_type = image.type.split("/")[-1]
  #     if image_type not in ['jpg','jpeg','png','jfif']:
  #       st.error("Invalid file type : {}".format(image.type), icon="🚨)")
  #     else:
  #       st.image(image, caption="User uploaded image")
  #       if image:
  #         user_image = Image.open(image)
  #         user_image.save(image_name)
  #         with st.spinner("Processing..."):
  #           image_features = featurization(image_name, convext_featurized_model)
  #           model_predict = cataract_model.predict(image_features)
  #           model_predict_proba = cataract_model.predict_proba(image_features)
  #           probability = model_predict_proba[0][model_predict[0]]
  #         st.write("Model Prediction:", model_predict)
  #         st.write("Model Prediction Probabilities:", model_predict_proba)
  #         col1, col2 = st.columns(2)
  #         with col1:
  #           st.header("Disease Type")
  #           st.subheader("{}".format(PREDICTION_LABELS[model_predict[0]]))
  #         with col2:
  #           st.header("Prediction Probability")
  #           st.subheader("{}".format(probability))

  # with tab2:
  #   cam_image = st.camera_input("Take a photo of the eye")
  #   if cam_image:
  #     st.image(image, caption="Captured image")
  #     user_image = Image.open(cam_image)
  #     user_image.save(image_name)
  #     with st.spinner("Processing..."):
  #           image_features = featurization(image_name, convext_featurized_model)
  #           model_predict = cataract_model.predict(image_features)
  #           model_predict_proba = cataract_model.predict_proba(image_features)
  #           probability = model_predict_proba[0][model_predict[0]]
  #     #st.write("Model Prediction:", model_predict)
  #     #st.write("Model Prediction Probabilities:", model_predict_proba)
  #     col1, col2 = st.columns(2)
  #     with col1:
  #           st.header("Disease Type")
  #           st.subheader("{}".format(PREDICTION_LABELS[model_predict[0]]))
  #     with col2:
  #           st.header("Prediction Probability")
  #           st.subheader("{}".format(probability))



