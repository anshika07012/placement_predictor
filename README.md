# placement_predictor
An End-to-End Machine Learning project predicting whether a student gets placed based on CGPA and IQ.  Tech Stack: Python, Pandas, Matplotlib, Scikit-Learn  Workflow: Data Cleaning, EDA (Scatter Plots), Feature Scaling, Train-Test Split, Classification Model Training &amp; Evaluation

📊 How the Model Works
Feature Inputs:

CGPA: Cumulative Grade Point Average (0.0 to 10.0)

IQ Score: Aptitude score (40 to 200)

Processing Pipeline:

Input feature values are scaled using scaler.pkl.

The scaled array is passed to the trained LogisticRegression model (model.pkl).

The model predicts the probability of placement (Placed vs. Not Placed).

Output:

Displays prediction result along with the model's confidence percentage.

# 🎓 Student Placement Predictor

An end-to-end Machine Learning web application that predicts a student's placement likelihood based on their academic performance (CGPA) and aptitude scores (IQ).

🔗 **Live Demo:** [https://placementpredictor-o3bip77b8pttv4o38oiuex.streamlit.app/](https://placementpredictor-o3bip77b8pttv4o38oiuex.streamlit.app/)

---

## 📌 Features

- **Interactive UI:** Simple and clean interface built with Streamlit.
- **Real-Time Predictions:** Instant calculation of placement outcome and model confidence percentage.
- **Robust Preprocessing:** Uses Scikit-Learn's `StandardScaler` to ensure features are normalized consistently with training data.
- **Cloud Deployed:** Automatically hosted and updated via Streamlit Community Cloud linked with GitHub.

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Machine Learning:** Scikit-Learn (Logistic Regression, StandardScaler)
- **Data Manipulation:** NumPy, Pandas
- **Web Framework:** Streamlit
- **Deployment:** Streamlit Cloud, Git, GitHub

---

## 📂 Project Structure

```text
placement_predictor/
│
├── app.py              # Streamlit web application script
├── model.pkl           # Trained Logistic Regression model
├── scaler.pkl          # Fitted StandardScaler object
├── requirements.txt    # Project dependencies for deployment
└── README.md           # Project documentation
