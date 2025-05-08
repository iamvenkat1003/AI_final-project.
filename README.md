# AI_final-project.


# 🧠 AI Vision Showdown: ConvNets vs Transformers

Welcome to **AI Vision Showdown**, an interactive web app where modern deep learning models compete head-to-head in computer vision tasks. Users upload images, select a task (Object Detection or Segmentation), and observe two powerful architectures—**ConvNets** and **Transformers**—battle it out with real-time visual results and performance scores.

---

## 📌 Project Overview

This project compares convolutional neural networks (CNNs) and transformer-based models for two major vision tasks:

- **Object Detection**: Identifying and classifying objects within an image
- **Instance Segmentation**: Detecting objects and outlining them with pixel-accurate masks

The app runs both models on the same image, displays side-by-side results with confidence scores, and allows users to **vote** on which model they believe performed better.

This demo encourages users to learn the practical differences between CNN and Transformer-based models through **engaging interaction** and **visual output comparison**.

---

## 🛠️ Tools, Models, and Frameworks Used

### 🔍 Object Detection:
- **ConvNet Model**: `torchvision.models.detection.fasterrcnn_resnet50_fpn`
- **Transformer Model**: [`facebook/detr-resnet-50`](https://huggingface.co/facebook/detr-resnet-50)

### 🖼️ Instance Segmentation:
- **ConvNet Model**: `torchvision.models.detection.maskrcnn_resnet50_fpn`
- **Transformer Model**: [`facebook/mask2former-swin-small-coco-instance`](https://huggingface.co/facebook/mask2former-swin-small-coco-instance)

### 🧰 Frameworks:
- [Gradio](https://www.gradio.app) – For building the interactive user interface
- [PyTorch](https://pytorch.org) – For model inference
- [Torchvision](https://pytorch.org/vision/stable/index.html) – For pretrained ConvNet models
- [Transformers](https://huggingface.co/docs/transformers/index) – For Hugging Face Transformer models
- [Matplotlib](https://matplotlib.org) – For visualization

---

## 🚀 How to Run the App Locally

### ✅ Install Requirements
pip install torch torchvision transformers gradio matplotlib


To Run the App
python app.py
Once launched, open your browser and go to http://127.0.0.1:7860 on the browser.

---

# 🌐 Live Demo
## 🎯 Try it now on Hugging Face Spaces
https://huggingface.co/spaces/JohnJoelMota/AI-FINAL-PROJECT

Note: It may take a few seconds to load initially.

###📊 Features
Upload your own images or try built-in examples

Choose between Object Detection or Segmentation

View ConvNet and Transformer model results side-by-side

Compare number of detected objects and confidence scores

Vote on which model performed better

Get visual and textual analysis of the results

---

## Project Structure
<pre> vision-showdown/ │ ├── app.py # Full Gradio web app ├── requirements.txt # Required Python packages ├── README.md # Project documentation (this file) └── TEST_IMG_x.jpg # Sample images used in the app </pre>

---

## Contributors

John Joel Mota

Venkateswarareddy Satti

Moditha Lekkala Chowdary

Nitesh Kumar Datta Vemanapalli

