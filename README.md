# fastapi-sklearn-logistic-regression-demo

![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

An end-to-end machine learning web application that analyzes medical insurance claim data. This project demonstrates the integration of a data science workflow into a robust web API using FastAPI. Users can upload a CSV file of claim data, and the application will automatically clean the data, train a Logistic Regression model to predict claim denial reasons, and return performance metrics and dynamic data visualizations.

## Table of Contents

- [Key Features](#key-features)
- [Screenshots](#screenshots)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Application](#running-the-application)
- [How to Use](#how-to-use)
- [Workflow Explanation](#workflow-explanation)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Contact](#contact)

---

## Key Features

- **CSV Data Ingestion:** Upload claim data directly through a user-friendly web interface.
- **Automated Data Preprocessing:** The backend automatically cleans currency fields (e.g., `'Payment Amount'`, `'Balance'`) and handles missing values.
- **Machine Learning Model Training:** A **Logistic Regression** model is trained on the fly to classify the reasons for claim denials based on features like `CPT Code` and `Balance`.
- **Dynamic Data Visualization:**
  - Generates a **Pair Plot** to visualize relationships between key features, colored by the denial reason.
  - Generates a **Decision Boundary Plot** to visually represent how the trained model separates the different classes.
- **Performance Metrics:** Calculates and displays the model's accuracy on a held-out test set.
- **RESTful API:** Built with FastAPI, providing a modern, high-performance, and well-documented API structure.
- **Modular Codebase:** The machine learning logic (preprocessing, training, visualization) is cleanly separated from the API logic for better maintainability.

## Screenshots

### **1. Upload Page**

The user is greeted with a simple interface to upload their CSV file.

![Homepage](assets/homepage.png?raw=true)

### **2. Results Page**

After processing, the application displays the model's accuracy and the generated plots.

![Homepage](assets/result.png?raw=true)

## Technology Stack

- **Backend:**
  - **FastAPI:** For building the high-performance, asynchronous web API.
  - **Uvicorn:** As the ASGI server to run the application.
  - **Jinja2:** For server-side rendering of HTML templates.
- **Data Science & Machine Learning:**
  - **Scikit-learn:** For model training (Logistic Regression), data splitting, and performance evaluation.
  - **Pandas:** For data manipulation and cleaning.
  - **NumPy:** For numerical operations, especially for generating the decision boundary grid.
- **Data Visualization:**
  - **Matplotlib & Seaborn:** For generating the pair plot and decision boundary plot.

## Project Structure

The project is organized with a clear separation of concerns, making it easy to navigate and extend.

```bash
/fastapi-claim-analyzer
|
├── ML/
│   ├── __init__.py
│   ├── preprocessing.py      # Functions for data loading and splitting
│   ├── train.py              # Functions for model training and prediction
│   └── visualization.py      # Functions for generating plots
│
├── templates/
│   ├── index.html            # HTML for the file upload form
│   └── result.html           # HTML to display the results
│
├── main.py                   # The main FastAPI application logic
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Setup and Installation

Follow these steps to get the project running locally.

### Prerequisites

- [Git](https://git-scm.com/)
- [Python 3.10+](https://www.python.org/downloads/)

### 1. Clone the Repository

```bash
git clone https://github.com/sajithsajith/fastapi-sklearn-logistic-regression-demo.git
cd fastapi-sklearn-logistic-regression-demo
```

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

- **On macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- **On Windows:**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```

### 3. Install Dependencies

Install all the required packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Running the Application

Once the setup is complete, you can run the FastAPI application using Uvicorn.

```bash
cd app
uvicorn main:app --reload
```

- The `--reload` flag enables hot-reloading, so the server will restart automatically after you make code changes.

The application will be available at **http://127.0.0.1:8000**.

## How to Use

1.  Open your web browser and navigate to `http://127.0.0.1:8000`.
2.  Click the **"Choose File"** button and select a CSV file containing medical claim data. A sample file can be used from the notebook you provided.
3.  Click the **"Upload and Analyze"** button.
4.  The application will process the file and display a results page with:
    - The model's accuracy percentage.
    - The data visualization pair plot.
    - The model's decision boundary plot.

## Workflow Explanation

The application follows a standard machine learning pipeline triggered by a file upload:

1.  **File Ingestion:** The `/upload` endpoint in `main.py` receives the uploaded CSV file.
2.  **Data Loading & Cleaning (`ML/preprocessing.py`):**
    - The raw file stream is read into a Pandas DataFrame.
    - Any rows with missing data after cleaning are dropped.
3.  **Data Splitting (`ML/preprocessing.py`):**
    - The target variable, `Denial Reason`, is label-encoded into numerical values.
    - The dataset is split into training (70%) and testing (30%) sets using `CPT Code` and `Balance` as features.
4.  **Model Training (`ML/train.py`):**
    - A `LogisticRegression` classifier is instantiated and trained on the training data (`X_train`, `y_train`).
5.  **Prediction & Evaluation (`ML/train.py`):**
    - The trained model makes predictions on the test set (`X_test`).
    - The `accuracy_score` is calculated by comparing the predictions with the actual test labels (`y_test`).
6.  **Visualization (`ML/visualization.py`):**
    - **Pair Plot:** A `seaborn.pairplot` is created to show pairwise relationships in the dataset, with points colored by the `Denial Reason`.
    - **Decision Boundary:** A mesh grid is created over the feature space of the test data. The model predicts the class for each point on the grid, which is then plotted as a colored contour map. The actual test data points are overlaid on this map.
    - Both plots are converted to a base64 encoded URI string so they can be embedded directly into the HTML response.
7.  **Response Rendering:** The final results (accuracy and plot URIs) are passed to the `result.html` Jinja2 template and rendered to the user.

## Future Improvements

- [ ] **Model Selection:** Add a dropdown to allow users to choose different classification models (e.g., SVM, Decision Tree, Random Forest).
- [ ] **Feature Selection:** Allow users to select which columns to use as features for training.
- [ ] **Error Handling:** Implement more granular error handling for malformed CSVs or missing columns.
- [ ] **CI/CD Pipeline:** Set up a GitHub Actions workflow to automatically test and deploy the application.
- [ ] **Dockerization:** Create a `Dockerfile` to containerize the application for easier deployment and scalability.
- [ ] **Frontend Framework:** Replace Jinja2 templates with a modern frontend framework like React or Vue.js for a more interactive user experience.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
