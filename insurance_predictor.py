import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Insights Studio", layout="wide")
st.title("ML Insights Studio - Healthcare Bundle")

# Sidebar navigation 
page = st.sidebar.selectbox("Select Page", ["Home", "Classification - Diabetes", "Regression - Insurance", "Data Explorer", "About"])

# 1. HOME
if page == "Home":
    st.markdown("Welcome to ML Insights Studio")
    st.write("Ye app 2 kaam karegi:")
    st.write("1. Classification: Diabetes Risk Prediction")
    st.write("2. Regression: Insurance Cost Prediction")
    st.info("Sidebar se koi bhi module select karo ")

# 2. CLASSIFICATION - DIABETES
elif page == "Classification - Diabetes":
    st.header("Diabetes Risk Prediction")

    uploaded_file = st.file_uploader("Diabetes CSV/Excel Upload karo", type=['csv', 'xlsx'], key="diabetes")

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("1. Data Display")
        st.dataframe(df.head())

        st.subheader("2. Visualization")
        col1, col2 = st.columns(2)
        fig1, ax1 = plt.subplots()
        sns.countplot(x='Outcome', data=df, ax=ax1)
        ax1.set_title("Diabetes Outcome Count")
        col1.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        sns.histplot(df['Glucose'], bins=30, kde=True, ax=ax2)
        ax2.set_title("Glucose Distribution")
        col2.pyplot(fig2)

        st.subheader("3. Model Training")
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
        col2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
        col3.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
        col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.3f}")

        st.subheader("4. Live Prediction")
        col1, col2, col3, col4 = st.columns(4)
        pregnancies = col1.number_input("Pregnancies", 0, 17, 3)
        glucose = col2.number_input("Glucose", 0, 200, 120)
        bp = col3.number_input("BloodPressure", 0, 122, 70)
        skin = col4.number_input("SkinThickness", 0, 99, 20)
        insulin = col1.number_input("Insulin", 0, 846, 80)
        bmi = col2.number_input("BMI", 0.0, 67.1, 25.0)
        dpf = col3.number_input("DiabetesPedigree", 0.0, 2.5, 0.5)
        age = col4.number_input("Age", 18, 81, 30)

        if st.button("Predict Diabetes"):
            input_data = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]], columns=X.columns)
            prediction = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]
            if prediction == 1:
                st.error(f"Result: Diabetic - Risk: {prob*100:.1f}%")
            else:
                st.success(f"Result: Non-Diabetic - Risk: {prob*100:.1f}%")
    else:
        st.warning("Pehle Diabetes ki CSV/Excel file upload karo")

# 3. REGRESSION - INSURANCE - 
elif page == "Regression - Insurance":
    st.header("Insurance Cost Prediction")

    uploaded_file = st.file_uploader("Insurance CSV Upload karo", type=['csv'], key="insurance")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("1. Data Display")
        st.dataframe(df.head())

        st.subheader("2. Visualization")
        col1, col2 = st.columns(2)
        fig1, ax1 = plt.subplots()
        sns.scatterplot(x=df['bmi'], y=df['charges'], hue=df['smoker'], ax=ax1)
        ax1.set_title("BMI vs Charges")
        col1.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        sns.histplot(df['charges'], bins=40, kde=True, ax=ax2)
        ax2.set_title("Charges Distribution")
        col2.pyplot(fig2)

        st.subheader("3. Model Training") 
        df_enc = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
        X = df_enc.drop('charges', axis=1)
        y = df_enc['charges']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"${mean_absolute_error(y_test, y_pred):.2f}")
        col2.metric("MSE", f"${mean_squared_error(y_test, y_pred):.2f}")
        col3.metric("RMSE", f"${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        col4.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")

        st.subheader("4. Live Prediction") 
        col1, col2 = st.columns(2)
        age = col1.slider("Age", 18, 64, 30)
        bmi = col2.slider("BMI", 15.0, 53.0, 25.0)
        children = col1.slider("Children", 0, 5, 1)
        sex = col2.selectbox("Sex", ["male", "female"])
        smoker = col1.selectbox("Smoker", ["no", "yes"])
        region = col2.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

        if st.button("Predict Cost"):
            input_df = pd.DataFrame([[age, bmi, children, sex, smoker, region]],
                                    columns=['age', 'bmi', 'children', 'sex', 'smoker', 'region'])
            input_enc = pd.get_dummies(input_df, columns=['sex', 'smoker', 'region'], drop_first=True)
            input_enc = input_enc.reindex(columns=X.columns, fill_value=0)
            pred = model.predict(input_enc)[0]
            st.success(f"Predicted Charges: ${pred:,.2f}")
    

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
    

# 5. ABOUT 
elif page == "About":
    st.header("About")
    st.write("Healthcare bundle: Diabetes + Insurance Cost Prediction")
    st.write("Tools: Streamlit, Scikit-learn, Pandas, Seaborn")
    st.write("Datasets: Medical Insurance + Pima Diabetes")
