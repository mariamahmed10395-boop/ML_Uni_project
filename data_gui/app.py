import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# --- Preprocessing & Modeling Imports ---
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')

# --- 1. Page Configuration ---
st.set_page_config(page_title="Weather Prediction App", page_icon="🌦️", layout="wide")

# --- 2. Helper Functions ---
def run_full_preprocessing(df, 
                           impute_method='KNN', 
                           outlier_method='Clipping', 
                           scaling_method='Standard', 
                           sampling_method='SMOTE',
                           use_pca=True):
    df_work = df.copy()
    
    num_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_work.select_dtypes(include=['object']).columns.tolist()
    
    # Impute categorical with mode
    cat_imp = SimpleImputer(strategy='most_frequent')
    df_work[cat_cols] = cat_imp.fit_transform(df_work[cat_cols])
    
    # Impute numerical
    if impute_method == 'KNN':
        num_imp = KNNImputer(n_neighbors=5)
    else:
        num_imp = SimpleImputer(strategy='median')
    df_work[num_cols] = num_imp.fit_transform(df_work[num_cols])

    # Handle outliers
    if outlier_method == 'Winsorization':
        for col in num_cols:
            df_work[col] = winsorize(df_work[col], limits=[0.05, 0.05])
    else:
        for col in num_cols:
            Q1, Q3 = df_work[col].quantile(0.25), df_work[col].quantile(0.75)
            IQR = Q3 - Q1
            df_work[col] = df_work[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

    # Encode categorical
    le = LabelEncoder()
    for col in cat_cols:
        if df_work[col].nunique() == 2:
            df_work[col] = le.fit_transform(df_work[col])
    
    remaining_cats = [c for c in cat_cols if df_work[c].nunique() > 2 and c not in ['Date', 'RainTomorrow']]
    df_work = pd.get_dummies(df_work, columns=remaining_cats, drop_first=True)

    # Split X and y
    X = df_work.drop(['RainTomorrow', 'Date'], axis=1, errors='ignore')
    y = df_work['RainTomorrow'] if 'RainTomorrow' in df_work.columns else None

    # Scale Data
    if scaling_method == 'Standard':
        scaler = StandardScaler()
    elif scaling_method == 'MinMax':
        scaler = MinMaxScaler()
    else:
        scaler = PowerTransformer()
    
    X_scaled = scaler.fit_transform(X)

    # Balance Data and PCA
    if y is not None:
        # Encode y if it's categorical
        if y.dtype == 'object':
            y = le.fit_transform(y)

        if sampling_method == 'SMOTE':
            sampler = SMOTE(random_state=42)
        else:
            sampler = RandomUnderSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X_scaled, y)
        
        if use_pca:
            pca = PCA(n_components=10)
            X_res = pca.fit_transform(X_res)
            
        return X_res, y_res
    
    return X_scaled, None

# --- 3. Main Sidebar Navigation ---
st.sidebar.title("📌 Project Workflow")
st.sidebar.info("Weather and Rain Prediction Project.")

page = st.sidebar.radio(
    "Navigate through project phases:",
    [
        "1️⃣ File Upload", 
        "2️⃣ Data Visualization", 
        "3️⃣ Preprocessing", 
        "4️⃣ Model Selection",
        "5️⃣ Model Evaluation"
    ]
)

# ==========================================
# Page 1: File Upload 
# ==========================================
if page == "1️⃣ File Upload":
    st.title("📂 File Upload")
    uploaded_file = st.file_uploader("Upload dataset file (CSV or Excel)", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success("File uploaded successfully! 🎉")
        
        col1, col2 = st.columns(2)
        with col1: st.metric("Number of Rows", df.shape[0])
        with col2: st.metric("Number of Columns", df.shape[1])
            
        st.dataframe(df.head())
        st.session_state['raw_data'] = df
    else:
        st.info("Waiting for data file to be uploaded...")

# ==========================================
# Page 2: Data Visualization
# ==========================================
elif page == "2️⃣ Data Visualization":
    st.title("📈 Weather Data Visualization")
    
    if 'raw_data' in st.session_state:
        df = st.session_state['raw_data'].copy()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Month'] = df['Date'].dt.month
            
        st.sidebar.markdown("---")
        st.sidebar.title("📊 Visualization Menu")
        plot_type = st.sidebar.radio("Click to display a plot:", ["1. Scatter Plot", "2. Box Plot", "3. Line Plot", "4. Count Plot"])
        
        if plot_type == "1. Scatter Plot":
            st.header("1. Scatter Plot: MinTemp vs MaxTemp")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sample_df = df.sample(n=min(5000, len(df)), random_state=42)
            sns.scatterplot(data=sample_df, x='MinTemp', y='MaxTemp', hue='RainTomorrow' if 'RainTomorrow' in df.columns else None, alpha=0.7, ax=ax1)
            st.pyplot(fig1)

        elif plot_type == "2. Box Plot":
            st.header("2. Box Plot: Humidity vs Rain Tomorrow")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            if 'Humidity3pm' in df.columns and 'RainTomorrow' in df.columns:
                sns.boxplot(data=df, x='RainTomorrow', y='Humidity3pm', palette='Set2', ax=ax2)
                st.pyplot(fig2)
            else:
                st.error("Required columns for Box Plot are missing.")

        elif plot_type == "3. Line Plot":
            st.header("3. Line Plot: Average Max Temp by Month")
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            if 'Month' in df.columns and 'MaxTemp' in df.columns:
                sns.lineplot(data=df, x='Month', y='MaxTemp', marker='o', color='coral', ax=ax3)
                plt.xticks(range(1, 13))
                st.pyplot(fig3)

        elif plot_type == "4. Count Plot":
            st.header("4. Count Plot: Rain Tomorrow Imbalance")
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            if 'RainTomorrow' in df.columns:
                sns.countplot(data=df, x='RainTomorrow', palette='pastel', ax=ax4)
                st.pyplot(fig4)
    else:
        st.warning("⚠️ Please upload the dataset from the first page first.")

# ==========================================
# Page 3: Preprocessing
# ==========================================
elif page == "3️⃣ Preprocessing":
    st.title("🧹 Data Preprocessing Pipeline")
    
    if 'raw_data' in st.session_state:
        df = st.session_state['raw_data'].copy()
        
        st.sidebar.markdown("---")
        st.sidebar.header("⚙️ Preprocessing Options")
        imp = st.sidebar.radio("1. Imputation Method", ["KNN", "Simple"])
        out = st.sidebar.selectbox("2. Outliers Handling", ["Clipping", "Winsorization"])
        sca = st.sidebar.selectbox("3. Scaling Method", ["Standard", "MinMax", "Power"])
        sam = st.sidebar.radio("4. Data Balancing", ["SMOTE", "UnderSampling"])
        pca_on = st.sidebar.checkbox("5. Apply PCA (10 Components)?", value=True)
        
        if st.button("Run Full Preprocessing Pipeline", type="primary"):
            with st.spinner('Applying preprocessing... ⏳'):
                X_final, y_final = run_full_preprocessing(df, imp, out, sca, sam, pca_on)
                
                # Split into Train and Test automatically for the user
                X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
                
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                
                st.success("Preprocessing & Splitting Completed! 🎉")
                st.metric("Total Samples after Balancing", len(y_final))
                st.dataframe(pd.DataFrame(X_train).head())
    else:
        st.warning("⚠️ Please upload the dataset first.")

# ==========================================
# Page 4: Model Selection
# ==========================================
elif page == "4️⃣ Model Selection":
    st.title("🤖 Model Selection & Training")
    
    # Try to get data from session state first, then from CSV
    if 'X_train' in st.session_state:
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        st.info("Using preprocessed data from Page 3.")
    else:
        try:
            X_train = pd.read_csv('X_train_Nada.csv')
            y_train = pd.read_csv('y_train_Nada.csv')
            if isinstance(y_train, pd.DataFrame): y_train = y_train.iloc[:, 0]
            st.success("Training data loaded from CSV files!")
        except:
            st.warning("⚠️ No data found. Please complete Page 3 or provide CSV files.")
            st.stop()

    st.subheader("Choose your ML Algorithm")
    algo = st.selectbox("Select Model:", ["Logistic Regression", "Random Forest", "Decision Tree", "SVM", "KNN"])

    if st.button("🚀 Train Model"):
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "SVM": SVC(probability=True),
            "KNN": KNeighborsClassifier()
        }
        
        model = models[algo]
        with st.spinner(f"Training {algo}..."):
            model.fit(X_train, y_train)
            st.session_state['trained_model'] = model
            st.session_state['model_name'] = algo
        st.success(f"✅ {algo} trained successfully! Go to the next page for Evaluation.")

# ==========================================
# Page 5: Model Evaluation
# ==========================================
elif page == "5️⃣ Model Evaluation":
    st.title("📊 Model Evaluation")

    if 'trained_model' not in st.session_state:
        st.warning("⚠️ Please train a model in Page 4 first!")
        st.stop()

    # Try to get test data
    if 'X_test' in st.session_state:
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
    else:
        try:
            X_test = pd.read_csv('X_test_Nada.csv')
            y_test = pd.read_csv('y_test_Nada.csv')
            if isinstance(y_test, pd.DataFrame): y_test = y_test.iloc[:, 0]
        except:
            st.error("❌ Test data not found!")
            st.stop()

    model = st.session_state['trained_model']
    algo_name = st.session_state['model_name']

    st.subheader(f"Results for: {algo_name}")

    if st.button("📊 Run Evaluation"):
        y_pred = model.predict(X_test)
        
        col1, col2 = st.columns(2)
        with col1:
            acc = accuracy_score(y_test, y_pred)
            st.metric("Test Accuracy", f"{acc*100:.2f}%")
            st.text("Detailed Report:")
            st.code(classification_report(y_test, y_pred))
            
        with col2:
            st.write("Confusion Matrix:")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)