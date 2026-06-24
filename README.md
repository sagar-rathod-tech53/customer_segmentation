# Customer Segmentation Using Unsupervised Learning

## Project Overview

This project performs Customer Segmentation using Machine Learning Unsupervised Learning techniques. The objective is to group customers with similar purchasing behavior into meaningful segments that can help businesses improve marketing strategies, customer retention, and sales performance.

The project uses clustering algorithms such as:

* K-Means Clustering
* Hierarchical Clustering
* DBSCAN

A Streamlit web application is provided to predict the customer segment based on customer purchasing information.

---

## Dataset

The dataset contains customer transaction information including:

* Invoice Number
* Stock Code
* Product Description
* Quantity
* Invoice Date
* Unit Price
* Customer ID
* Country

Feature engineering is applied to generate:

* Revenue = Quantity × Unit Price
* Encoded Country Information

---

## Project Phases

### Phase 1: Exploratory Data Analysis (EDA)

* Dataset Overview
* Missing Value Analysis
* Duplicate Record Analysis
* Statistical Summary
* Correlation Matrix
* Outlier Detection
* Country-wise Analysis
* Revenue Analysis

### Phase 2: Data Preprocessing

* Data Cleaning
* Label Encoding
* One-Hot Encoding
* Feature Engineering
* Feature Scaling using StandardScaler

### Phase 3: Machine Learning Models

Implemented clustering algorithms:

#### K-Means Clustering

* Elbow Method
* Cluster Formation

#### Hierarchical Clustering

* Agglomerative Clustering
* Dendrogram Analysis

#### DBSCAN

* Density-Based Clustering
* Noise Detection

### Phase 4: Model Evaluation

Evaluation Metrics:

* Silhouette Score
* Davies-Bouldin Index
* Calinski-Harabasz Score

### Phase 5: Streamlit Deployment

Interactive web application that:

* Accepts customer details
* Performs preprocessing
* Predicts customer segment
* Displays cluster information

---

## Project Structure

customer_segmentation/

├── dataset/

│ └── Online Retail.xlsx

├── notebooks/

│ ├── 01_EDA.ipynb

│ ├── 02_Preprocessing.ipynb

│ ├── 03_Clustering.ipynb

│ └── 04_Evaluation.ipynb

├── models/

│ ├── kmeans_model.pkl

│ ├── hierarchical_model.pkl

│ ├── dbscan_model.pkl

│ ├── scaler.pkl

│ ├── columns.pkl

│ └── country_encoder.pkl

├── app.py

├── train.py

├── requirements.txt

└── README.md

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sagar-rathod-tech53/customer_segmentation.git
cd customer_segmentation
```

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Model Training

Run the training script:

```bash
python train.py
```

This will:

* Load dataset
* Clean data
* Encode categorical features
* Scale features
* Train clustering models
* Save models in the models folder

---

## Model Evaluation

Run the evaluation notebook:

```bash
04_Evaluation.ipynb
```

Evaluation Metrics:

* Silhouette Score
* Davies-Bouldin Index
* Calinski-Harabasz Score

The best-performing clustering algorithm is selected based on these metrics.

---

## Run Streamlit Application

Start the application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Joblib
* OpenPyXL
* Streamlit

---

## Future Enhancements

* RFM Customer Segmentation
* Customer Lifetime Value Analysis
* PCA Visualization
* UMAP Visualization
* Automated Cluster Naming
* Real-Time Customer Segmentation API

---

## Author

Customer Segmentation Project using Machine Learning Unsupervised Learning Algorithms and Streamlit Deployment.
