import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Insights Studio", layout="wide")
st.title("ML Insights Studio - Education Bundle")

# Sidebar navigation 
page = st.sidebar.selectbox("Select Page", ["Home", "Classification - Pass/Fail", "Regression - Exam Score", "Data Explorer", "About"])

# 1. HOME
if page == "Home":
    st.markdown("Welcome to ML Insights Studio")
    st.write("Ye app 2 kaam karegi:")
    st.write("1. Classification: Student Pass/Fail Prediction")
    st.write("2. Regression: Student Exam Score Prediction")
    st.info("Sidebar se koi bhi module select karo")
    
    st.subheader("Default Dataset Preview")
    try:
        df = pd.read_csv("student_exam_scores.csv")
        st.dataframe(df.head())
        st.write("**Shape:**", df.shape)
    except:
        st.warning("student_exam_scores.csv file upload karo ya same folder me rakho")

# 2. CLASSIFICATION - PASS/FAIL
elif page == "Classification - Pass/Fail":
    st.header("Student Pass/Fail Prediction")
    
    uploaded_file = st.file_uploader("Student CSV/Excel Upload Karo", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # 33 passing marks se Pass/Fail column banao
        if 'Result' not in df.columns:
            df['Result'] = df['exam_score'].apply(lambda x: 'Pass' if x >= 33 else 'Fail')
        
        st.subheader("1. Data Display")
        st.dataframe(df.head())
        st.write("**Class Distribution:**")
        st.write(df['Result'].value_counts())
        
        # 2. Visualization
        st.subheader("2. Visualization")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='Result', palette='Set2', ax=ax)
            ax.set_title("Pass vs Fail Count")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='Result', y='hours_studied', palette='Set1', ax=ax)
            ax.set_title("Study Hours vs Result")
            st.pyplot(fig)
        
        # 3. Model Training
        X = df[['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores']]
        y = df['Result']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        st.subheader("3. Model Evaluation")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
        col2.metric("Precision", f"{precision_score(y_test, y_pred, pos_label='Pass'):.3f}")
        col3.metric("Recall", f"{recall_score(y_test, y_pred, pos_label='Pass'):.3f}")
        col4.metric("F1 Score", f"{f1_score(y_test, y_pred, pos_label='Pass'):.3f}")
        
        st.write("**Confusion Matrix:**")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'], ax=ax)
        st.pyplot(fig)
        
        # 4. Live Prediction 
        st.subheader("4. Live Prediction")
        col1, col2, col3, col4 = st.columns(4)
        hours_studied = col1.number_input("Hours Studied", 0.0, 12.0, 5.0, 0.1)
        sleep_hours = col2.number_input("Sleep Hours", 0.0, 12.0, 7.0, 0.1)
        attendance = col3.number_input("Attendance %", 0.0, 100.0, 75.0, 0.1)
        prev_score = col4.number_input("Previous Scores", 0.0, 100.0, 60.0, 0.1)
        
        if st.button("Predict Result"):
            input_data = pd.DataFrame([[hours_studied, sleep_hours, attendance, prev_score]], 
                                    columns=['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores'])
            prediction = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0]
            if prediction == 'Pass':
                st.success(f"Result: Pass - Confidence: {prob[1]*100:.1f}%")
            else:
                st.error(f"Result: Fail - Confidence: {prob[0]*100:.1f}%")
    else:
        st.warning("Pehle Student ki CSV/Excel file upload karo")

# 3. REGRESSION - EXAM SCORE
elif page == "Regression - Exam Score":
    st.header("Student Exam Score Prediction")
    
    uploaded_file = st.file_uploader("Student CSV Upload Karo", type=['csv'], key='reg')
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("1. Data Display")
        st.dataframe(df.head())
        st.write("**Statistical Summary:**")
        st.write(df.describe())
        
        # 2. Visualization
        st.subheader("2. Visualization")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='hours_studied', y='exam_score', hue='attendance_percent', palette='viridis', ax=ax)
            ax.set_title("Study Hours vs Exam Score")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.histplot(df['exam_score'], kde=True, bins=20, color='skyblue', ax=ax)
            ax.set_title("Distribution of Exam Scores")
            st.pyplot(fig)
        
        # 3. Model Training
        X = df[['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores']]
        y = df['exam_score']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        st.subheader("3. Model Evaluation")
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("MSE", f"{mse:.2f}")
        col3.metric("RMSE", f"{rmse:.2f}")
        col4.metric("R² Score", f"{r2:.3f}")
        
        # 4. Live Prediction 
        st.subheader("4. Live Prediction")
        col1, col2 = st.columns(2)
        hours = col1.slider("Hours Studied", 0.0, 12.0, 5.0, 0.1)
        sleep = col2.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.1)
        attendance = col1.slider("Attendance %", 0.0, 100.0, 75.0, 0.1)
        prev_score = col2.slider("Previous Score", 0.0, 100.0, 60.0, 0.1)
        
        if st.button("Predict Score"):
            input_df = pd.DataFrame([[hours, sleep, attendance, prev_score]], 
                                  columns=['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores'])
            pred = model.predict(input_df)[0]
            st.success(f"Predicted Exam Score: {pred:.2f} / 100")
            
            if pred >= 90:
                st.balloons()
                st.write("Grade: **A+** Outstanding!")
            elif pred >= 75:
                st.write("Grade: **A** Excellent!")
            elif pred >= 60:
                st.write("Grade: **B** Good!")
            elif pred >= 33:
                st.write("Grade: **C** Pass")
            else:
                st.write("Grade: **F** Need Improvement")
    else:
        st.warning("Pehle Student ki CSV file upload karo")

# 4. DATA EXPLORER
elif page == "Data Explorer":
    st.header("Data Explorer")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.dataframe(df.head())
        st.write(df.describe())
        
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# 5. ABOUT
elif page == "About":
    st.header("About")
    st.write("**Education Bundle: Student Pass/Fail + Exam Score Prediction**")
    st.write("**Tools Used:** Streamlit, Scikit-learn, Pandas, Seaborn, Matplotlib")
    st.write("**Dataset:** Student Exam Scores Dataset")
    st.write("**Features:** hours_studied, sleep_hours, attendance_percent, previous_score")
    st.write("**Target 1:** Result - Pass/Fail for Classification")
    st.write("**Target 2:** exam_score - 0 to 100 for Regression")
    st.write("**Model:** RandomForestClassifier & RandomForestRegressor")
    st.info("Developed for Education Bundle Capstone Project")
