# 🚆 Automatic Defect Detection in Railway Tracks Using CNN & Transfer Learning

Timely and accurate railway track inspection is vital for safe train operations. This project leverages deep learning to automate defect detection in railway track images, classifying them as **Defective** or **Non-Defective** using two approaches:

- A **Custom Convolutional Neural Network (CNN)** built from scratch  
- A **Transfer Learning model** based on VGG16 pre-trained on ImageNet  

---

## 📌 Project Highlights

- **Dataset**: Labeled images of railway tracks categorized as *Defective* and *Non-Defective*
- **Image Preprocessing**: Resizing, normalization, and data augmentation for robustness
- **Models**:
  - Custom CNN with 3 Conv blocks and dropout for regularization
  - Transfer Learning with VGG16 + custom dense layers
- **Evaluation Metrics**: Accuracy, Precision, Recall, and F1-score
- **Deployment**: Streamlit web app for real-time image classification

---

## 🧰 Tools & Technologies Used

| Category         | Tools / Libraries                               |
|------------------|--------------------------------------------------|
| **Programming**  | Python 3.8+                                      |
| **Deep Learning**| TensorFlow, Keras                                |
| **Modeling**     | Custom CNN, VGG16 Transfer Learning              |
| **Visualization**| Matplotlib, Seaborn                              |
| **App Interface**| Streamlit                                        |
| **Data Handling**| NumPy                                    |
| **Metrics**      | scikit-learn (Precision, Recall, F1-score)       |
| **Deployment**   | Localhost (Cloud Ready: Streamlit Cloud, AWS)    |

---

## 🗂️ Dataset Structure
```plaintext
dataset/
├── Train/
│   ├── Defective/
│   └── Non_defective/
├── Validation/
│   ├── Defective/
│   └── Non_defective/
└── Test/
    ├── Defective/
    └── Non_defective/
```
---

## 🧠 Model Performance

| Model        | Accuracy | Precision | Recall | F1-score |
|--------------|----------|-----------|--------|----------|
| Custom CNN   | 86.36%   | 87%       | 86%    | 86%      |
| VGG16 (TL)   | **90.91%** | **91%**     | **91%**  | **91%**    |

**Observation**: VGG16 demonstrated better generalization and faster convergence.

---

## 🚀 Streamlit Web App

An interactive Streamlit web app allows users to:

- Upload railway track images  
- Select model (Custom CNN or VGG16)  
- View predictions with confidence score and visualization  

**Run locally**:
```bash
streamlit run Detect.py
```
---
## 📌 Conclusion

This project demonstrates the feasibility and effectiveness of deep learning—especially transfer learning—in automating defect detection for railway infrastructure.

### **Key Insights**

- ✅ **VGG16-based transfer learning** provided higher accuracy and faster convergence than the custom CNN.  
- 🔍 **Image preprocessing** and **data augmentation** significantly improved model generalization.  
- 🖥️ The **Streamlit interface** enables real-time, user-friendly inspection capabilities.  

**✅ These methods can reduce inspection time, minimize manual errors, and significantly enhance railway safety monitoring systems.**


