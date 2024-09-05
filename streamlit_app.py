import streamlit as st
import torch
from PIL import Image
from prediction import pred_class

def calculate_ctr(image: Image.Image) -> float:
    # Placeholder for CTR calculation logic
    # Replace this with actual logic to compute CTR
    return 0.60  # Example value, replace with actual calculation logic

st.title('Cardiovascular Disease Prediction')
st.header('Please Upload a Chest X-ray Image')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the model and handle potential errors
try:
    model = torch.load('mobilenetv3_large_100_checkpoint_fold1.pt', map_location=device)
except FileNotFoundError:
    st.error("Model file not found. Please check the file path.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# Upload image
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Predict'):
        # Calculate CTR
        ctr = calculate_ctr(image)
        st.write(f"## Calculated CTR: {ctr:.2f}")
