# 🩸 Blood Cell Detection Project  
**Detecting Red Blood Cells (RBC), White Blood Cells (WBC), and Platelets in Microscopic Images**

This project uses **Faster R-CNN with ResNet50 FPN** to detect blood cells from microscopic images. It supports:
- Training on Pascal VOC-style dataset (`RBC`, `WBC`, `Platelet`)
- Flask API for real-time inference
- HTML interface for uploading and visualizing results
- Color-coded bounding boxes per class


## 📁 Folder Structure
```bash 
Blood-Cells-Detector-Project/
│
├── app.py # Flask API code
├── utils.py # Model loading, prediction & visualization
├── Blood_Cells_Detection_Module.pth # Trained PyTorch model
├── index.html # Frontend: image upload + result display
├── README.md # This file
├── requirements.txt # Python dependencies
└── data/ # Dataset folder
   ├── Training/
   │ ├── Images/
   │ └── Annotations/
   ├── Validation/
   │ ├── Images/
   │ └── Annotations/
   └── Testing/
   │ ├── Images/
   │ └── Annotations/
```

---

## 🧠 Features

| Feature | Description |
|--------|-------------|
| 🔍 Object Detection | Detects 3 types of blood cells: RBC, WBC, Platelet |
| 🧪 Pretrained Model | Uses Faster R-CNN with ResNet50 FPN |
| 🎨 Class-wise Bounding Boxes | Each class has its own color: red=RBC, blue=WBC, green=Platelet |
| 🌐 REST API | Built with Flask to accept `.png/.jpg` files and return predictions |
| 💻 Local Inference | Works fully offline after setup |
| 🖼️ Visualization | Returns base64 image with drawn boxes |

---

## 🛠️ Requirements
  - torch==2.0+
  - torchvision==0.15+
  - flask
  - pillow
  - matplotlib
  - numpy<2.0


# 🚀 How to Run Locally

## ⚠️ Note About the Trained Model
The trained PyTorch model (`Blood_Cells_Detector_Model.pth`) is stored using **Git LFS**.
To use it:
1. Make sure Git LFS is installed locally:
   ```bash
   git lfs install
   
## 1. Clone the Repo
```bash
   git clone https://github.com/Hatem-Alathwari/Blood-CellsDetector-Project.git
   cd Blood-Cells-Detector-Project
```

## 2.Install dependencies:
```bash
pip install torch torchvision flask pillow matplotlib numpy
⚠️ On Windows, avoid NumPy 2.x – use pip install "numpy<2" 
```
## 3.Start Flask API
```bash
  python app.py
```
## 4.Open browser at:
```bash
  http://localhost:5000/
```


## 👤 Author
  **Hatem Alathwari**
  * https://github.com/Hatem-Alathwari

## 🙌 Acknowledgments
  * TorchVision Models
  *  Flask Documentation
  *  Matplotlib
  * Blood cell dataset (custom Pascal VOC format)

## 🧾 License
  * MIT License – see LICENSE file.
