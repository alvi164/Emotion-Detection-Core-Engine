# Real-Time Emotion Detection Engine with Dependency Alignment

A robust, real-time facial emotion recognition pipeline built using Deep Learning (CNN) and OpenCV. This project features a complete machine learning workflow, from local dataset training to real-time webcam inference, with carefully aligned software dependencies for optimal GPU/CPU execution.

## 🚀 Key Features
- **Real-Time Inference:** High-frame-rate facial emotion detection via webcam streaming.
- **Custom Trained Architecture:** Includes localized training scripts utilizing Convolutional Neural Networks (CNNs).
- **Multi-Class Classification:** Robustly detects primary human emotions including Sad, Surprise, Happy, Angry, and Neutral.
- **Dependency Optimization:** Fully aligned environment configurations to prevent library conflicts (such as Keras/TensorFlow and OpenCV mismatches).

---

## 📂 Project Directory Structure

```text
Emotion-Detection-Core-Engine/
│
├── data/
│   ├── train/               # Organized dataset for training (e.g., surprise, sad, etc.)
│   └── test/                # Organized dataset for validation/testing
│
├── emotion_model.h5         # Locally trained Deep Learning weights file
├── train_model.py           # Script to train the CNN architecture from scratch
├── final_detection.py       # Core engine for real-time facial expression tracking
├── test_webcam.py           # Diagnostic script to test OpenCV camera stream
└── README.md                # Project documentation

```
🛠️ Requirements & Installation
To run this core engine seamlessly on your local environment (optimized for Windows/Linux with dedicated GPU capabilities), follow these steps:

1. Clone the Repository
Bash
```text
git clone [https://github.com/alvi164/Emotion-Detection-Core-Engine.git](https://github.com/alvi164/Emotion-Detection-Core-Engine.git)
cd Emotion-Detection-Core-Engine
```
3. Setup Isolated Environment
It is highly recommended to use a virtual or Conda environment to preserve strict dependency versions:

Bash
```text
conda create -n emotion_env python=3.10 -y
conda activate emotion_env
```
3. Core Dependencies
Install the strictly aligned packages to ensure tensorflow and opencv-python operate without matrix dimension errors:

Bash
```text
pip install tensorflow
pip install opencv-python
pip install numpy matplotlib
```
💻 How to Use
Step 1: Model Training (Optional)
If you want to re-train the core neural network model using your local image dataset inside the data/ directory, run:

Bash
```text
python train_model.py
```
Step 2: Run Real-Time Detection
To launch the webcam interface and evaluate real-time facial responses via the optimized emotion_model.h5 weights, run:

Bash
```text
python final_detection.py
```
Press 'q' to safely exit the video stream window.

🤝 Author & Contributions
Developed with a focus on stable machine learning deployment pipelines by [Syed Alvi (alvi164)](https://github.com/alvi164)
Feel free to open issues or submit Pull Requests to optimize the network layers or extend detection metrics!
```text
