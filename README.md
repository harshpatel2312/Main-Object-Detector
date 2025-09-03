# Main Object Detector 🖼️🔍

This project uses **YOLO (You Only Look Once)** to detect objects in an image and intelligently select the **main object** based on a custom scoring function that balances:
- 📏 **Object Size** (bounding box area)
- ✅ **Model Confidence** (YOLO’s confidence score)
- 🎯 **Center Proximity** (how close the object is to the image center)
  
The selected object is highlighted with a bounding box, and the **Region of Interest (ROI)** can be cropped for further use.



## 🚀 Features
- Detects multiple objects in an image with **YOLOv11n**
- Calculates a **main object score** = α·area + β·confidence – γ·center distance
- Draws a bounding box only around the chosen **main object**
- Crops and displays the detected ROI separately
- Modular structure:
  - `MOD.py` → main object selection logic
  - `demo.py` → runnable demo with visualization



## 📂 Project Structure
```
Main-Object-Detector/
│── images/                # sample images
│── MOD.py                 # main object selection function
│── demo.py                # run script to test on image
│── requirements.txt       # dependencies
```



## 📦 Requirements
- Python 3.8+
- OpenCV
- NumPy
- Ultralytics (YOLO)
- PyTorch



## ⚙️ Installation
Clone the repo:
```bash
git clone https://github.com/harshpatel2312/Main-Object-Detector.git
cd Main-Object-Detector
```

Create virtual environment (Optional, but recommended):
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

Install requirements:
```bash
pip install -r requirements.txt
```



## ▶️ Usage
Run the demo script on a test image:
```python
python demo.py
```

Steps performed:
- Loads YOLOv11n model
- Detects all objects in `images/your_img.jpeg`
- Selects the main object
- Displays:
  - Original image
  - Image with bounding box on main object
  - Cropped ROI of the main object
 


## 🧮 Main Object Scoring
Each detected object is scored by:
```
score ​= α⋅area ​+ β⋅conf ​− γ⋅center_dist
```

Where:
- `area` = normalized box area (relative to image size)
- `conf` = YOLO’s confidence score
- `center_dist` = normalized distance from image center

Default weights:
- α = 0.5 (area)
- β = 0.4 (confidence)
- γ = 0.1 (center distance penalty)



## 🖼️ Example Output
![Example](https://github.com/user-attachments/assets/ff8f3f14-0f1d-4514-98f0-e9ae740a9f06)



## 📌 Future Improvements
- Extend to video streams (track main object over time)
- Support YOLO segmentation for finer ROI extraction
- Experiment with different α, β, γ scoring strategies



## 👨‍💻 Developer
Developed by **Harsh Patel**
