
# Fake Job Posting Detection ML

## Overview
Fake Job Posting Detection ML is a machine learning project that identifies whether a job posting is **real or fake**. 
The system uses a **Random Forest Classifier** and provides a **Tkinter GUI dashboard** for loading datasets, preprocessing data, training the model, visualizing analytics, and predicting fraudulent job postings.

This project was developed as a **B.Tech Final Year Project**.

## Features
- Load training and testing datasets (CSV)
- Data preprocessing and feature engineering
- Machine learning model training (Random Forest)
- Fraud prediction for job postings
- Data visualization (bar plot, box plot, correlation heatmap)
- Interactive Tkinter GUI dashboard

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Tkinter

## Project Structure
```
fake-job-posting-detection-ml
│
├── fake_job_code.py
├── train.csv
├── test.csv
├── README.md
└── requirements.txt
```

## Installation

### 1. Install Python
Download Python from:
https://www.python.org/downloads/

### 2. Install Required Libraries
Run the following command:

```
pip install pandas numpy scikit-learn matplotlib seaborn
```

## How to Run

1. Open terminal or VS Code
2. Navigate to the project folder

```
cd Desktop
cd "4th year project"
cd "fake job.4th year Project"
cd "fake job"
```

3. Run the program

```
python fake_job_code.py
```

## Model
The project uses **Random Forest Classifier**, an ensemble learning algorithm that builds multiple decision trees and combines their predictions to improve accuracy.

## Output
The system predicts whether a job posting is:
- **REAL**
- **FAKE**

It also displays probability scores for each prediction.

## Author
**Talari Charandas**  
Email: nanitalari21@gmail.com
