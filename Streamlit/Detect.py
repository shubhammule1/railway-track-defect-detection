import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Class labels
class_names = ['Defective', 'Non Defective']

st.title('Rail Track Fault Detection')
st.write("Upload an image to check if it's Defective or Non Defective.")

# Step 1: Upload the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Step 2: Proceed if image is uploaded
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Step 3: Ask user to choose model AFTER uploading image
    model_choice = st.selectbox(
        "Select Model for Prediction",
        ("Choose a model", "CNN Model", "Transfer Learning Model")
    )

    # Step 4: Perform prediction only if a valid model is selected
    if model_choice != "Choose a model":
        
        # Load selected model
        @st.cache_resource
        def load_selected_model(choice):
            if choice == "CNN Model":
                return load_model('my_model.h5')
            else:
                return load_model('my_model_TL.h5')

        model = load_selected_model(model_choice)

        # Set appropriate image size
        target_size = (150, 150) if model_choice == "CNN Model" else (224, 224)

        # Prediction function
        def predict_image(img, model, target_size):
            img = img.resize(target_size)
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)

            if prediction.shape[1] == 1:  # Sigmoid
                prob = prediction[0][0]
                predicted_class = class_names[int(prob > 0.5)]
                confidence = prob if prob > 0.5 else 1 - prob
            else:  # Softmax
                class_index = np.argmax(prediction)
                predicted_class = class_names[class_index]
                confidence = prediction[0][class_index]

            return predicted_class, confidence

        # Step 5: Predict and display results
        st.write("Classifying...")
        predicted_class, confidence = predict_image(img, model, target_size)

        st.write(f"### Model Used: {model_choice}")
        st.write(f"### Prediction: {predicted_class}")
        st.write(f"### Confidence: {confidence:.2%}")
