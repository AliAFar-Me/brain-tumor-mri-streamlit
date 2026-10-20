# brain-tumor-mri-streamlit
Interactive Streamlit web application for 4-class Brain Tumor MRI classification (Glioma, Meningioma, Pituitary, No Tumor) powered by a fine-tuned Vision Transformer and hosted via Hugging Face Hub.

# 🧠 Brain Tumor MRI Classification — Streamlit Web App

An interactive clinical decision-support demonstration built with **Streamlit**, **PyTorch**, and **Hugging Face Transformers**. This application hosts the best-performing Vision Transformer from the **AIxMed 2026 Bootcamp (Project 2)** to classify human brain MRI scans into four diagnostic categories: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**.

## 🚀 Live Demo & Model Weights
* **Live Streamlit Web App:** [[Insert your `.streamlit.app` URL here]](https://brain-tumor-mri-app-wnxsj73nbqgprvktsgtkgn.streamlit.app/)
* **Hugging Face Model Hub:** [AliAFar/brain-tumor-vit-best](https://huggingface.co/AliAFar/brain-tumor-vit-best)
* **Dataset:** [Hemg/Brain-Tumor-MRI-Dataset](https://huggingface.co/datasets/Hemg/Brain-Tumor-MRI-Dataset) (7,023 brain MRI scans across 4 classes)

## 🏗️ Project Overview
In **Project 2 (Vision Transformers)**, four state-of-the-art Vision Transformer backbones were benchmarked across validation loss curves and multi-class evaluation metrics:
* **Supervised Models:** `google/vit-base-patch16-224` and `microsoft/swin-tiny-patch4-window7-224`
* **Self-Supervised Models:** `microsoft/rad-dino` (DINOv2 radiology encoder) and `facebook/vit-mae-base` (Masked Autoencoder)

The best-performing model on the validation split (`microsoft/rad-dino` with a custom linear classification head) was evaluated on the hold-out test set and exported to the Hugging Face Hub for real-time inference in this application.

## 💻 Running Locally
```bash
git clone [https://github.com/YOUR-USERNAME/brain-tumor-mri-streamlit.git](https://github.com/YOUR-USERNAME/brain-tumor-mri-streamlit.git)
cd brain-tumor-mri-streamlit
pip install -r requirements.txt
streamlit run app.py
