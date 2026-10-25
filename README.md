# 🧠 Brain Tumor MRI Classification with Vision Transformers

An end-to-end medical imaging pipeline and interactive **Streamlit** web application classifying human brain MRI scans into four diagnostic categories: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**. Developed as the final project for the **AI x Med 2026 Bootcamp**.

🔗 **Live Links:**
* **🚀 Live Streamlit Web App:**  [`.streamlit.app`](https://brain-tumor-mri-app-wnxsj73nbqgprvktsgtkgn.streamlit.app/)
* **🤗 Hugging Face Model Hub:** [AliAFar/brain-tumor-vit-best](https://huggingface.co/AliAFar/brain-tumor-vit-best)
* **📂 Dataset:** [Hemg/Brain-Tumor-MRI-Dataset](https://huggingface.co/datasets/Hemg/Brain-Tumor-MRI-Dataset)

---
## 🔬 Architectures Benchmarked (Supervised vs. Self-Supervised)
Four Hugging Face Vision Transformer backbones were trained and compared across 15 epochs:
* **Supervised Models:** `google/vit-base-patch16-224` (`Base_ViT`) & `microsoft/swin-tiny-patch4-window7-224` (`Swin_ViT`)
* **Self-Supervised Models:** `microsoft/rad-dino` (`RAD-DINO`) & `facebook/vit-mae-base` (`MAE_ViT`)

### 1. Training Loss & Validation Macro-AUROC
Across 15 epochs, `Base_ViT` achieved the lowest training loss (~0.22) and highest validation Macro-AUROC (~0.989), followed by `RAD-DINO` (~0.979), `Swin_ViT` (~0.970), and `MAE_ViT` (~0.953):

![Training Loss and Validation Macro-AUROC](Training%20and%20AUROC%20Curves.png)

### 2. Validation F1-Score Performance
`Base_ViT` reached a peak validation F1-score of ~0.912, outperforming the self-supervised `RAD-DINO` runner-up (~0.880):

![F1 Validation Performance](F1%20Validation%20Performance%20AX.png)

### 3. Hold-Out Test Set Confusion Matrix
Evaluating `Base_ViT` on the 1,600-image unseen test set achieved **87.13% overall accuracy** (1,394 / 1,600), with **98.0% recall on No Tumor** (392 / 400) and **96.75% recall on Pituitary** (387 / 400):

![Test Set Confusion Matrix](Confusion%20Matrix%20VBase.png)

---

## ⚙️ Cloud Architecture & Local Setup
To keep deployment lightweight on CPU servers, the frozen `google/vit-base-patch16-224` backbone is paired dynamically at runtime with the custom linear classification head (`custom_classifier_head.pt`) hosted on the Hugging Face Hub.

```bash
git clone [https://github.com/YOUR-USERNAME/brain-tumor-mri-streamlit.git](https://github.com/YOUR-USERNAME/brain-tumor-mri-streamlit.git)
cd brain-tumor-mri-streamlit
pip install -r requirements.txt
streamlit run app.py
