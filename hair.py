import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# Load class names
vitamin_deficiencies = ["Vitamin A", "Vitamin B", "Vitamin C", "Vitamin D", "Vitamin E"]  # Adjusted to match trained model
num_classes = len(vitamin_deficiencies)

# Define the CNN Model
class CustomCNN(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load trained model
model = CustomCNN(num_classes=num_classes)
model.fc2 = nn.Linear(512, num_classes)  # Ensure correct output size
model.load_state_dict(torch.load('custom_cnn_4_classes.pth', map_location=torch.device('cpu')), strict=False)
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Streamlit UI
st.set_page_config(page_title="Vitamin Deficiency Detector", layout="wide")
st.title("🍊 Vitamin Deficiency Detector")
st.write("Upload three images to analyze and predict possible vitamin deficiencies.")

# Sidebar Information
st.sidebar.header("Model Information")
st.sidebar.write(f"**Detected Deficiencies:** {num_classes}")
st.sidebar.write("This model is trained to classify images related to different vitamin deficiencies.")

# Upload multiple images
uploaded_files = st.file_uploader("Upload 3 images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) == 3:
    results = []
    col1, col2, col3 = st.columns(3)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100
            _, predicted = torch.max(output, 1)
            predicted_class = vitamin_deficiencies[predicted.item()]
        
        results.append((predicted_class, probabilities))
        
        # Display image in columns
        with [col1, col2, col3][idx]:
            st.image(image, caption=f"Uploaded Image {idx+1}", use_column_width=True)
            st.success(f"### Predicted Deficiency: {predicted_class}")
            st.write("### Confidence Scores:")
            for i, prob in enumerate(probabilities):
                st.write(f"{vitamin_deficiencies[i]} Deficiency: {prob:.2f}%")
    
    # Determine the most likely deficiency across all images
    deficiency_scores = torch.zeros(num_classes)
    for _, probs in results:
        deficiency_scores += probs
    
    most_likely_deficiency = vitamin_deficiencies[torch.argmax(deficiency_scores).item()]
    st.markdown("---")
    st.success(f"### Overall Most Likely Deficiency: {most_likely_deficiency}")

# Footer
st.markdown("---")
st.markdown("Developed by **Surgical AI Team** 🚀")
