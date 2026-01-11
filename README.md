# 🤟 VocalHands

### Real-Time ASL Sign Language Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.8+-green.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-0.10+-orange.svg" alt="MediaPipe">
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3+-red.svg" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-ff4b4b.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<p align="center">
  <b>Breaking communication barriers through computer vision and machine learning</b>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [Web Deployment](#-web-deployment)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🎯 Overview

**VocalHands** is an AI-powered sign language detection system that translates American Sign Language (ASL) hand gestures into text in real-time. Using computer vision and machine learning, the system captures hand movements through a webcam, processes them using MediaPipe's hand landmark detection, and classifies the signs using a K-Nearest Neighbors (KNN) algorithm.

### The Problem

Over **466 million people** worldwide have disabling hearing loss. Communication between deaf/hard-of-hearing individuals and those who don't know sign language remains a significant challenge in daily interactions, healthcare, education, and emergency situations.

### Our Solution

VocalHands provides an accessible, real-time translation tool that:

- Requires only a standard webcam
- Works offline (no internet required for detection)
- Achieves **98.5% accuracy** on ASL alphabet recognition
- Builds sentences automatically from detected signs
- Can be deployed as a web application for broader accessibility

---

## 🎬 Demo

### Desktop Application

```
python detect_signs.py
```

- Real-time webcam detection
- Live prediction display with confidence scores
- Automatic sentence building
- FPS counter for performance monitoring

### Web Application

```
streamlit run streamlit_app.py
```

- Browser-based detection
- No installation required for end users
- Mobile-friendly interface
- Shareable via URL

---

## ✨ Features

| Feature                 | Description                                              |
| ----------------------- | -------------------------------------------------------- |
| **Real-Time Detection** | Process video at 30+ FPS for instant feedback            |
| **High Accuracy**       | 98.5% test accuracy on 28 sign classes                   |
| **Sentence Building**   | Automatically builds words/sentences from detected signs |
| **Calibration Tool**    | Personalize the model with your own hand samples         |
| **Web Deployment**      | Streamlit-based web app for easy sharing                 |
| **Cross-Platform**      | Works on Windows, macOS, and Linux                       |
| **Offline Support**     | No internet required after installation                  |
| **Open Source**         | Fully customizable and extendable                        |

---

## 🔬 How It Works

### System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Webcam    │ ──► │  MediaPipe   │ ──► │     KNN     │ ──► │   Display    │
│   Input     │     │  Hand Detect │     │  Classifier │     │   Output     │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
      │                    │                    │                    │
      ▼                    ▼                    ▼                    ▼
   BGR Image         21 Landmarks         Prediction          Sign + Sentence
   (1280x720)        (63 features)        + Confidence        on Screen
```

### Step-by-Step Process

#### 1. Hand Detection (MediaPipe)

MediaPipe Hands detects 21 3D landmarks on each hand:

```
         8   12  16  20       ◄── Fingertips
         │   │   │   │
     7   11  15  19  │        ◄── DIP Joints
     │   │   │   │   │
     6   10  14  18  │        ◄── PIP Joints
     │   │   │   │   │
     5───9───13──17──┤        ◄── MCP Joints
          \         │
           \        │
        4   \       │         ◄── Thumb
        │    \      │
        3     \     │
        │      \    │
        2       \   │
        │        \  │
        1─────────0─┘         ◄── Wrist
              WRIST
```

#### 2. Feature Extraction

Each landmark provides (x, y, z) coordinates:

- **21 landmarks × 3 coordinates = 63 features per hand**

#### 3. Normalization

Features are normalized for:

- **Position invariance**: Centered around wrist (landmark 0)
- **Scale invariance**: Normalized by palm size (wrist to middle finger MCP distance)

```python
# Normalization algorithm
coords_centered = coords - wrist_position
palm_size = distance(wrist, middle_finger_mcp)
coords_normalized = coords_centered / palm_size
```

#### 4. Classification (KNN)

K-Nearest Neighbors algorithm:

- Stores all training samples in memory
- For new input, finds K closest training samples
- Returns majority class among neighbors
- Distance-weighted voting for better accuracy

#### 5. Temporal Smoothing

To reduce noise and flickering:

- Maintains a buffer of recent predictions
- Uses majority voting over 5 frames
- Requires consistent predictions before adding to sentence

---

## 🛠 Technology Stack

| Component            | Technology      | Purpose                            |
| -------------------- | --------------- | ---------------------------------- |
| **Hand Detection**   | MediaPipe Hands | 21-point hand landmark detection   |
| **Image Processing** | OpenCV          | Webcam capture, image manipulation |
| **Machine Learning** | Scikit-Learn    | KNN classifier, preprocessing      |
| **Data Processing**  | NumPy           | Numerical computations             |
| **Web Framework**    | Streamlit       | Web application deployment         |
| **Language**         | Python 3.8+     | Core development                   |

### Why These Technologies?

- **MediaPipe**: Google's production-grade ML solution, optimized for real-time on-device inference
- **KNN**: Simple, interpretable, no training time, easy to update with new samples
- **Streamlit**: Rapid web app development with built-in camera support

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- pip (Python package manager)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/VocalHands.git
cd VocalHands

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

---

## 🚀 Usage

### Option 1: Use Pre-Trained Model

If a trained model is included:

```bash
python detect_signs.py
```

### Option 2: Train Your Own Model

#### Step 1: Get Training Data

**Using Kaggle ASL Dataset (Recommended):**

```bash
# Download from: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
# Extract to VocalHands/data/ folder
python process_kaggle_dataset.py --max_samples 1500
```

**Or collect your own data:**

```bash
python collect_data.py
```

#### Step 2: Train the Model

```bash
python train_model.py
```

#### Step 3: Run Detection

```bash
python detect_signs.py
```

### Option 3: Calibrate for Your Hands

Improve accuracy by adding your personal hand samples:

```bash
python calibrate.py
```

### Controls

| Key         | Action                |
| ----------- | --------------------- |
| `Q` / `ESC` | Quit application      |
| `C`         | Clear sentence        |
| `SPACE`     | Add space to sentence |
| `BACKSPACE` | Delete last character |

---

## 📊 Dataset

### Kaggle ASL Alphabet Dataset

- **Source**: [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Total Images**: 87,000+
- **Classes**: 29 (A-Z + space, delete, nothing)
- **Images per Class**: ~3,000
- **Image Size**: 200×200 pixels

### Processed Dataset

After MediaPipe processing:

| Metric                  | Value                     |
| ----------------------- | ------------------------- |
| **Total Samples**       | 36,756                    |
| **Features per Sample** | 63                        |
| **Success Rate**        | 83.3%                     |
| **Signs Supported**     | 28 (A-Z + SPACE + DELETE) |

### Data Processing Pipeline

```
Raw Images (87K) → MediaPipe Detection → Landmark Extraction → Normalization → NumPy Arrays
```

---

## 📈 Model Performance

### Training Results

| Metric                     | Value      |
| -------------------------- | ---------- |
| **Test Accuracy**          | **98.39%** |
| **Training Samples**       | 29,404     |
| **Test Samples**           | 7,352      |
| **Best K Value**           | 1          |
| **Cross-Validation Score** | 98.04%     |

### Per-Class Performance

| Sign      | Precision | Recall | F1-Score |
| --------- | --------- | ------ | -------- |
| A-Z (avg) | 98%       | 98%    | 98%      |
| SPACE     | 100%      | 99%    | 99%      |
| DELETE    | 98%       | 100%   | 99%      |

### Challenging Signs

Some signs with similar hand shapes show lower accuracy:

- **M/N**: 90-92% (similar finger positions)
- **U/V/R**: 94-95% (similar extended fingers)

### Improving Accuracy

1. **Calibration**: Add personal hand samples
2. **More Data**: Process more Kaggle images
3. **Better Lighting**: Ensure good hand visibility
4. **Consistent Position**: Keep hand centered in frame

---

## 🌐 Web Deployment

### Streamlit Cloud (Free)

1. Push code to GitHub
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Deploy with one click!

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

### Running Locally

```bash
pip install streamlit
streamlit run streamlit_app.py
```

Access at: `http://localhost:8501`

---

## 📁 Project Structure

```
VocalHands/
│
├── 📱 Applications
│   ├── detect_signs.py          # Desktop real-time detection
│   ├── streamlit_app.py         # Web application
│   ├── collect_data.py          # Manual data collection
│   └── calibrate.py             # Personal calibration tool
│
├── 🧠 Model & Training
│   ├── train_model.py           # Model training script
│   ├── process_kaggle_dataset.py # Dataset processor
│   └── models/
│       └── knn_sign_model.pkl   # Trained model
│
├── 🔧 Utilities
│   ├── config.py                # Configuration settings
│   └── utils/
│       ├── __init__.py
│       └── hand_detector.py     # MediaPipe wrapper
│
├── 📊 Data
│   ├── data/                    # Raw Kaggle dataset
│   └── dataset/                 # Processed landmarks
│       ├── A/ ... Z/
│       ├── SPACE/
│       └── DELETE/
│
├── 📝 Documentation
│   ├── README.md                # This file
│   └── DEPLOYMENT.md            # Deployment guide
│
└── ⚙️ Configuration
    ├── requirements.txt         # Desktop dependencies
    ├── requirements_streamlit.txt # Web dependencies
    └── .streamlit/config.toml   # Streamlit settings
```

---

## 🔮 Future Improvements

### Short Term

- [ ] Add support for dynamic gestures (J, Z with motion)
- [ ] Implement word prediction/autocomplete
- [ ] Add text-to-speech output
- [ ] Support for two-handed signs

### Medium Term

- [ ] Train with deep learning (CNN/LSTM) for better accuracy
- [ ] Add support for other sign languages (BSL, ISL, etc.)
- [ ] Mobile app development (iOS/Android)
- [ ] Real-time language translation

### Long Term

- [ ] Full sentence/phrase recognition
- [ ] Bidirectional translation (text to sign)
- [ ] AR/VR integration
- [ ] Edge device deployment (Raspberry Pi)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Areas for Contribution

- Adding support for more sign languages
- Improving model accuracy
- UI/UX enhancements
- Documentation improvements
- Bug fixes

---

## 🙏 Acknowledgments

- **[MediaPipe](https://google.github.io/mediapipe/)** - Google's ML framework for hand detection
- **[Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)** - Training data
- **[Scikit-Learn](https://scikit-learn.org/)** - Machine learning library
- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[OpenCV](https://opencv.org/)** - Computer vision library

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**

- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/your-profile)

---

<p align="center">
  <b>Made with ❤️ for accessibility and inclusion</b>
</p>

<p align="center">
  🤟 Breaking barriers, one sign at a time 🤟
</p>
