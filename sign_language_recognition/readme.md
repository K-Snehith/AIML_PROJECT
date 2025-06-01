# Real-Time Sign Language Recognition

This project implements a real-time sign language recognition system using Python, OpenCV, MediaPipe, and TensorFlow. It recognizes American Sign Language (ASL) static hand gestures (A–Z, plus 'space', 'del', and 'nothing') and converts them into corresponding text in real time.

## 📌 Features

- Real-time recognition using webcam input  
- Hand tracking using MediaPipe  
- CNN-based classification trained on the ASL Alphabet Dataset  
- Supports 29 classes (A–Z, space, del, nothing)  
- Displays predictions live on the video feed  
- Optionally forms words from letters and adds text-to-speech output  

##  Technologies Used

- Python 3.9  
- OpenCV  
- MediaPipe  
- TensorFlow & Keras  
- NumPy  
- Scikit-learn  
- pyttsx3 (for text-to-speech, optional)

##  Dataset

- **ASL Alphabet Dataset**  
- Training Folder: `asl_alphabet_train`  
- Testing Folder: `asl_alphabet_test`  
- Each class contains 3000+ labeled images (200x200 pixels)

 Train the Model


sign_language_model.h5 (model)

classes.npy (label names)

Run Real-Time Prediction

python realtime_predict.py
A webcam window will open and show real-time predictions of hand gestures.


📊 Results
Training Accuracy: ~99.8%

Validation Accuracy: ~98.8%

Real-time performance with minimal latency

