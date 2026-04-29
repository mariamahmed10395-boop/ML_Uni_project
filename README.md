# Weather Prediction Machine Learning App 🌦️

This project provides an end-to-end interactive Machine Learning pipeline for weather prediction (e.g., predicting "Rain Tomorrow") using a Streamlit graphical user interface. It allows users to upload their dataset, visualize data distributions, perform extensive data preprocessing, and train & evaluate various machine learning models all within a web browser.

## Features

- **1️⃣ File Upload**: Upload your dataset in CSV or Excel format.
- **2️⃣ Data Visualization**: Interactive plots including Scatter Plots, Box Plots, Line Plots, and Count Plots to understand the data distribution and target variable imbalance.
- **3️⃣ Data Preprocessing Pipeline**: Highly customizable preprocessing steps including:
  - **Imputation**: KNN Imputer, Simple Imputer (Median/Most Frequent)
  - **Outlier Handling**: Clipping, Winsorization
  - **Scaling**: StandardScaler, MinMaxScaler, PowerTransformer
  - **Data Balancing**: SMOTE (Over-sampling), Random Under-sampling
  - **Dimensionality Reduction**: Principal Component Analysis (PCA)
- **4️⃣ Model Selection & Training**: Train popular classification algorithms:
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
- **5️⃣ Model Evaluation**: Evaluate trained models with metrics like Accuracy, Detailed Classification Reports, and visual Confusion Matrices.

## Project Structure

- `data_gui/app.py`: The main Streamlit application script containing the UI and pipeline logic.
- `data_preprocessing/data_pre.ipynb`: Jupyter notebook containing data exploration and preprocessing experiments.
- `Data/`: Directory for storing datasets (e.g., `weatherAUS.csv`).

## Installation and Setup

1. **Navigate to the project directory**:
   ```bash
   cd "c:\Users\zbook g6\Uni_project\ML_Uni_project"
   ```

2. **Install required dependencies**:
   Ensure you have Python installed. You will need the following libraries:
   ```bash
   pip install streamlit pandas numpy seaborn matplotlib scikit-learn imbalanced-learn scipy openpyxl
   ```

3. **Run the Streamlit application**:
   Navigate to the GUI directory and start the app:
   ```bash
   cd data_gui
   streamlit run app.py
   ```

## Usage

1. Open your browser to the local URL provided by Streamlit (usually `http://localhost:8501`).
2. Use the sidebar to navigate through the 5 phases of the project workflow.
3. Start by uploading your dataset (e.g., the `weatherAUS.csv` file from the `Data/` folder) on the first page.
4. Proceed through visualization, choose your preprocessing parameters, select a model to train, and view its evaluation metrics.