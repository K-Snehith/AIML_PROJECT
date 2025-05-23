# Movie Recommendation System – Content-Based Filtering

This project implements a content-based movie recommender system that suggests movies based on genre similarity. It uses TF-IDF vectorization and cosine similarity, and features a simple, interactive UI built with Streamlit.

---

## Project Objective

To build a machine learning-based application that recommends similar movies based on user selection by analyzing movie genre metadata.

---

## Tools and Technologies

- Python  
- Pandas  
- Scikit-learn  
- Streamlit  
- MovieLens Dataset (`movies.csv`)

---

## Dataset Summary

The system utilizes the `movies.csv` file from the MovieLens dataset. This file includes:
- Movie titles  
- Genre information  

---

## Workflow Overview

### 1. Data Preprocessing
- Cleaned genre text by replacing pipe symbols (`|`) with spaces.
- Applied TF-IDF vectorization to convert genre text into numerical feature vectors.

### 2. Similarity Computation
- Calculated cosine similarity between all movie vectors.
- Selected the top 5 most similar movies for any given title using similarity scores.

---

## User Interface

The front end is developed using Streamlit and includes the following features:
- A dropdown menu to select a movie.
- A “Recommend” button to generate suggestions.
- Output: Top 5 genre-based recommended movies.

---

## Features

- Efficient content-based filtering  
- Lightweight and easy to deploy  
- Minimal and user-friendly UI  

---

## How to Run the Application

1. Install required packages:
   ```bash
   pip install pandas scikit-learn streamlit
streamlit run app.py
