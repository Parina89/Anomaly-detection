import streamlit as st
import io
#import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from utils.dataloader import get_train_test_loaders
from utils.model import CustomVGG


from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Disable scientific notation
np.set_printoptions(suppress=True)

st.set_page_config(page_title="InspectorsAlly", page_icon=":camera:")
st.title("InspectorsAlly")

st.caption(
    "Boost Your Quality Control with InspectorsAlly - The Ultimate AI-Powered Inspection App"
)
st.write(
    "Try clicking a product image and watch how an AI Model will classify it between Good / Anomaly."
)

# Fix path with raw string
with st.sidebar:
    img = Image.open(r"C:\Users\Haha CORPORATION\OneDrive\Desktop\week-13\Dependencies\InspectorsAlly - Anomaly Detection\docs\overview_dataset.jpg")
    st.image(img)
    st.subheader("About InspectorsAlly")
    st.write(
        "InspectorsAlly is a powerful AI-powered application designed to help businesses streamline their quality control inspections."
    )

# Image input
def load_uploaded_image(file):
    return Image.open(file)

st.subheader("Select Image Input Method")
input_method = st.radio("options", ["File Uploader", "Camera Input"], label_visibility="collapsed")

uploaded_file_img, camera_file_img = None, None

if input_method == "File Uploader":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        uploaded_file_img = load_uploaded_image(uploaded_file)
        st.image(uploaded_file_img, caption="Uploaded Image", width=300)
        st.success("Image uploaded successfully!")
elif input_method == "Camera Input":
    st.warning("Please allow access to your camera.")
    camera_image_file = st.camera_input("Click an Image")
    if camera_image_file:
        camera_file_img = load_uploaded_image(camera_image_file)
        st.image(camera_file_img, caption="Camera Input Image", width=300)
        st.success("Image clicked successfully!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use raw string
data_folder = r"C:\Users\Haha CORPORATION\OneDrive\Desktop\week-13\Dependencies\InspectorsAlly - Anomaly Detection\data"
subset_name = "leather"
data_folder = os.path.join(data_folder, subset_name)

# Model inference function
def Anomaly_Detection(pil_image, root):
    """
    Perform anomaly detection on a PIL image.
    """
    model_path = r"C:\Users\Haha CORPORATION\OneDrive\Desktop\week-13\Dependencies\InspectorsAlly - Anomaly Detection\weights\leather_model.h5"
    model = torch.load(model_path, map_location=device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.sigmoid(output).cpu().numpy().squeeze()

    # Set class based on threshold
    predicted_class = "Good" if probs < 0.5 else "Anomaly"

    if predicted_class == "Good":
        return " Your product has been classified as 'Good' with no anomalies detected."
    else:
        return " Anomaly detected in the product. Please inspect it further."

#  Submit button handler
if st.button(label="Submit a Leather Product Image"):
    st.subheader("Output")
    if input_method == "File Uploader" and uploaded_file_img:
        selected_img = uploaded_file_img
    elif input_method == "Camera Input" and camera_file_img:
        selected_img = camera_file_img
    else:
        st.warning("Please upload or click an image.")
        selected_img = None

    if selected_img:
        with st.spinner("Analyzing image..."):
            prediction = Anomaly_Detection(selected_img, data_folder)
            st.success(prediction)
