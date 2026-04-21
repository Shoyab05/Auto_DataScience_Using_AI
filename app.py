import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, r2_score, mean_squared_error

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

st.set_page_config(page_title="Auto Data Science Using AI", layout="wide")
st.title("🤖 Auto Data Science Using AI")

file = st.file_uploader("📂 Upload CSV")

if file:
    df = pd.read_csv(file)

    # =========================
    # DATA PREVIEW
    # =========================
    st.subheader("📊 Raw Data")
    st.write(df.head())

    st.subheader("🔍 Data Info")
    st.write("Shape:", df.shape)
    st.write("Missing:\n", df.isnull().sum())
    st.write("Duplicates:", df.duplicated().sum())

    # =========================
    # CLEANING
    # =========================
    st.subheader("🧹 Data Cleaning")

    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    df.drop_duplicates(inplace=True)
    st.success("✅ Cleaning Done")

    # =========================
    # EDA
    # =========================
    st.subheader("📈 EDA")

    num_df = df.select_dtypes(include=np.number)

    if not num_df.empty:

        st.write("Statistical Summary")
        st.write(num_df.describe())

        # Histogram (only 5 cols)
        st.write("Histogram (Top Features)")
        cols = num_df.columns[:5]
        for col in cols:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.hist(num_df[col], bins=30)
            ax.set_title(col)
            st.pyplot(fig)

        # Correlation (clean)
        st.write("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(num_df.corr(), cmap="coolwarm", annot=False, ax=ax)
        st.pyplot(fig)

    else:
        st.warning("No numeric data")

    # =========================
    # FEATURE ENGINEERING
    # =========================
    st.subheader("⚙️ Feature Engineering")

    drop_cols = [col for col in df.columns if df[col].nunique() == 1]

    if drop_cols:
        st.write("Removed:", drop_cols)
        df.drop(drop_cols, axis=1, inplace=True)
    else:
        st.write("No unnecessary columns")

    # =========================
    # ENCODING
    # =========================
    st.subheader("🔄 Encoding")

    le = LabelEncoder()
    for col in df.select_dtypes(include='object'):
        df[col] = le.fit_transform(df[col])

    st.success("Encoding Done")

    # =========================
    # TARGET
    # =========================
    target = st.selectbox("🎯 Select Target", df.columns)

    if target:

        X = df.drop(target, axis=1)
        y = df[target]

        # Problem detection
        problem = "classification" if y.nunique() <= 10 else "regression"
        st.info(f"Problem Type: {problem}")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # =========================
        # MODELS
        # =========================
        if problem == "classification":
            models = {
                "Logistic": LogisticRegression(max_iter=1000, class_weight='balanced'),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(class_weight='balanced'),
                "Extra Tree": DecisionTreeClassifier(max_depth=5)
            }
        else:
            models = {
                "Linear": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Extra Tree": DecisionTreeRegressor(max_depth=5)
            }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            if problem == "classification":
                score = accuracy_score(y_test, pred)
            else:
                score = r2_score(y_test, pred)

            results[name] = score

        st.subheader("📊 Model Results")
        st.write(results)

        best_model_name = max(results, key=results.get)
        best_model = models[best_model_name]

        st.success(f"🏆 Best Model: {best_model_name}")

        # =========================
        # EVALUATION
        # =========================
        pred = best_model.predict(X_test)

        if problem == "classification":
            st.write("Accuracy:", accuracy_score(y_test, pred))
            st.write("Precision:", precision_score(y_test, pred, average='weighted'))
            st.write("Recall:", recall_score(y_test, pred, average='weighted'))

            cm = confusion_matrix(y_test, pred)

            fig, ax = plt.subplots(figsize=(4,3))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                cbar=False,
                annot_kws={"size": 10},
                ax=ax
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")

            st.pyplot(fig)

        else:
            st.write("R2 Score:", r2_score(y_test, pred))
            st.write("MSE:", mean_squared_error(y_test, pred))

        # =========================
        # SAVE MODEL
        # =========================
        joblib.dump(best_model, "model.pkl")
        st.success("💾 Model Saved")

        # =========================
        # PREDICTION
        # =========================
        st.subheader("🔮 Prediction")

        input_data = []
        for col in X.columns:
            val = st.number_input(f"{col}", value=0.0)
            input_data.append(val)

        if st.button("Predict"):
            arr = np.array(input_data).reshape(1, -1)
            arr = scaler.transform(arr)
            pred = best_model.predict(arr)

            st.success(f"Prediction: {pred[0]}")