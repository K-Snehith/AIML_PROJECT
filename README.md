

This repository includes two machine learning projects:

1. **Real-Time Sign Language Recognition** – A real-time ASL recognition system using webcam and deep learning.
2. **Movie Recommendation System** – A content-based filtering application that recommends movies by genre similarity.

---

## 📘 1. Real-Time Sign Language Recognition

This project implements a real-time sign language recognition system using Python, OpenCV, MediaPipe, and TensorFlow. It recognizes static ASL hand gestures (A–Z, plus 'space', 'del', and 'nothing') and converts them into corresponding text in real-time.

###  Features

- Real-time recognition using webcam input
- Hand tracking using MediaPipe
- CNN-based classification trained on ASL Alphabet Dataset
- Supports 29 classes (A–Z, space, del, nothing)
- Displays predictions live on the video feed
- Optionally forms words and outputs them via text-to-speech

###  Technologies Used

- Python 3.9
- OpenCV
- MediaPipe
- TensorFlow + Keras
- NumPy, Scikit-learn
- pyttsx3 (optional for TTS)

### 🗂 Dataset

- **ASL Alphabet Dataset**
- Training Folder: `asl_alphabet_train`
- Testing Folder: `asl_alphabet_test`
- Each class has 3000+ labeled images

### ⚙️ How to Run

1. Install dependencies:
```bash
pip install opencv-python mediapipe tensorflow numpy scikit-learn pyttsx3
```

2. Train the model:
```bash
python train_model.py
```

3. Run real-time prediction:
```bash
python realtime_predict.py
```

### 📊 Results

- Training Accuracy: ~99.8%
- Validation Accuracy: ~98.8%
- Real-time performance with minimal latency

---

## 🎬 2. Movie Recommendation System – Content-Based Filtering

A content-based recommender system that suggests movies based on genre similarity using TF-IDF and cosine similarity. It features an interactive UI built with Streamlit.

###  Objective

To build an application that recommends similar movies by analyzing genre metadata.

###  Tools and Technologies

- Python
- Pandas
- Scikit-learn
- Streamlit
- MovieLens Dataset (`movies.csv`)

###  Dataset Summary

- Uses `movies.csv` from the MovieLens dataset
- Contains movie titles and genre metadata

###  Workflow Overview

1. **Data Preprocessing**
   - Cleaned genre strings (`|` → space)
   - Applied TF-IDF vectorization

2. **Similarity Computation**
   - Cosine similarity between movie vectors
   - Retrieved top 5 most similar movies

### User Interface

Built with **Streamlit**:
- Dropdown to select a movie
- "Recommend" button to fetch suggestions
- Displays top 5 genre-based recommendations

### 🚀 How to Run the Application

1. Install requirements:
```bash
pip install pandas scikit-learn streamlit
```

2. Run the app:
```bash
streamlit run app.py
```

###  Features

- Fast, efficient recommendations
- Minimal and user-friendly interface
- Easy to deploy via Streamlit

