import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from transformers import ViTImageProcessor, ViTModel
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
    "or **No Tumor** using a fine-tuned **ViT-Base (`google/vit-base-patch16-224`)** Vision Transformer."
)

REPO_ID = "AliAFar/brain-tumor-vit-best"
BACKBONE_ID = "google/vit-base-patch16-224"
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# 2. Define the Model Wrapper
class Classifier(nn.Module):
    def __init__(self, backbone, num_classes=4):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        # Extract the [CLS] token representation directly from last_hidden_state
        # (Avoids ViTModel's randomly initialized pooler_output!)
        cls_features = outputs.last_hidden_state[:, 0]
        return self.head(cls_features)

# 3. Cached Model Loader
@st.cache_resource
def load_custom_model():
    # Load processor and pretrained ViT-Base backbone
    processor = ViTImageProcessor.from_pretrained(BACKBONE_ID)
    backbone = ViTModel.from_pretrained(BACKBONE_ID)

    # Download the lightweight custom head weights from your HF repo
    weights_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="custom_classifier_head.pt",
        force_download=True  # Ensures Streamlit fetches your latest uploaded .pt file
    )

    model = Classifier(backbone, num_classes=len(CLASSES))
    raw_state = torch.load(weights_path, map_location=torch.device("cpu"))

    # Automatically map the linear weight [4, 768] and bias [4] regardless of layer attribute name in Colab
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

with st.spinner("Loading ViT-Base (`google/vit-base-patch16-224`) and custom head..."):
    processor, model = load_custom_model()

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
