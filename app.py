import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoModel, AutoImageProcessor
from huggingface_hub import hf_hub_download
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="Brain Tumor MRI Classifier — AIxMed 2026",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor MRI Classification")
st.markdown(
    "Upload a brain MRI scan to classify it into **Glioma**, **Meningioma**, **Pituitary**, "
    "or **No Tumor** using a fine-tuned **RAD-DINO** Vision Transformer."
)

REPO_ID = "AliAFar/brain-tumor-vit-best"
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# 2. Define the Model Wrapper
class Classifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        features = (
            outputs.pooler_output
            if getattr(outputs, "pooler_output", None) is not None
            else outputs.last_hidden_state[:, 0]
        )
        return self.head(features)

# 3. Cached Model Loader (Downloads once on startup)
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(REPO_ID)
    backbone = AutoModel.from_pretrained(REPO_ID)
    
    weights_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="custom_classifier_head.pt"
    )
    
    model = Classifier(backbone, num_classes=len(CLASSES))
    raw_state = torch.load(weights_path, map_location=torch.device("cpu"))
    
    # Automatically match the linear weight [4, 768] and bias [4] regardless of key name
    head_weight = next(
        v for v in raw_state.values() if v.ndim == 2 and v.shape[0] == len(CLASSES)
    )
    head_bias = next(
        v for v in raw_state.values() if v.ndim == 1 and v.shape[0] == len(CLASSES)
    )
    model.head.weight.data.copy_(head_weight)
    model.head.bias.data.copy_(head_bias)
    
    model.eval()
    return processor, model

with st.spinner("Loading RAD-DINO Vision Transformer from Hugging Face..."):
    processor, model = load_model()

# 4. File Uploader & Inference UI
uploaded_file = st.file_uploader("Upload an MRI Scan (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Brain MRI", use_container_width=True)

    if st.button("Run Diagnostic Prediction", type="primary"):
        with st.spinner("Running inference..."):
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                logits = model(inputs["pixel_values"])
                probs = torch.softmax(logits, dim=-1).squeeze().tolist()

            pred_idx = int(torch.argmax(logits, dim=-1).item())
            pred_label = CLASSES[pred_idx]
            confidence = probs[pred_idx] * 100

            st.success(f"### Predicted Class: **{pred_label}** ({confidence:.2f}% confidence)")

            # Display probability distribution across all 4 classes
            prob_df = pd.DataFrame({
                "Tumor Class": CLASSES,
                "Probability (%)": [round(p * 100, 2) for p in probs]
            }).set_index("Tumor Class")
            st.bar_chart(prob_df)
