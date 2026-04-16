# 🌍 Land Use Change Detection with SAM & U-Net

## 📌 Overview

This project focuses on detecting **land use and land cover changes** from satellite imagery using deep learning techniques. It combines **U-Net for segmentation** and **Segment Anything Model (SAM)** for enhanced region understanding.

The system analyzes input satellite images and identifies changes such as:

* Urban expansion 🏙️
* Deforestation 🌳
* Water body variations 💧

---

## 🚀 Features

* 🧠 Deep Learning-based change detection
* 🛰️ Works on satellite imagery
* 🎯 Accurate segmentation using U-Net
* 🔍 Enhanced detection using SAM
* 🌐 Simple frontend interface for testing
* ⚡ API-based backend using Python

---

## 🛠️ Tech Stack

**Frontend**

*React JavaScript

**Backend**

* Python (Flask)

**Machine Learning**

* U-Net (for segmentation)
* SAM (Segment Anything Model)

**Tools & Libraries**

* TensorFlow / Keras
* PyTorch
* OpenCV
* NumPy

---

## 📂 Project Structure

```
project/
│
├── app.py                  # Backend API
├── sam.py                  # SAM integration
├── train_unet.py           # Model training script
├── test_api.py             # API testing
├── test_change_detection.py
├── index.html              # Frontend
├── requirements.txt        # Dependencies
├── package.json            # Frontend dependencies
├── API_DOCS.md             # API documentation
└── CHANGE_DETECTION_README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/Greeshma1803/major.git
cd major
```

---

### 2️⃣ Create virtual environment

```
python -m venv .venv
.venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Run the application

```
python app.py
```

---

## 🧪 Usage

1. Upload satellite images
2. Run change detection
3. View segmented output highlighting changes

---

## 📊 Results

* Successfully detects land use changes
* Provides segmented output maps
* Helps in environmental monitoring and urban planning

<img width="1246" height="624" alt="image" src="https://github.com/user-attachments/assets/1326199a-d232-4999-8b92-ffe1ffa2685f" />
Fig: Dashboard showing change detection output by manually uploading satellite data of two different time periods


<img width="1259" height="630" alt="image" src="https://github.com/user-attachments/assets/39aa30a3-e974-429a-9e2a-8067f3a6d7d4" />
Fig: U-Net Change Detection Output by taking image directly from NASA GIBS (Long:26.6139, Lat:77.2090, Date1:05-04-2026 ,Date2:11-04-2026)


<img width="1278" height="639" alt="image" src="https://github.com/user-attachments/assets/4618d533-6214-444c-80b4-d210f253f19f" />
Fig: Classification Output on segmented image



---

## 📥 Model Weights

Due to large file size, model weights are not included.

---

## ⚠️ Limitations

* Requires good quality satellite images
* Model performance depends on training data
* High computation for large images

---

## 🔮 Future Scope

* Real-time satellite data integration
* Improved accuracy using advanced models
* Deployment as a web application
* Integration with GIS systems


