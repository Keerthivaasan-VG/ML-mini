import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👨‍💼",
    layout="wide"
)

st.title("Employee Attrition Prediction")
st.write(
    "Predict whether an employee is likely to **leave (1)** or **stay (0)** "
    "using a Random Forest classification model."
)

@st.cache_resource
def train_model():
    df = pd.read_csv("employee_attrition_dataset.csv")

    X = df.drop(columns=["employee_id", "left"])
    y = df["left"]

    X = pd.get_dummies(
        X,
        columns=["salary"],
        drop_first=False
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return model, X.columns.tolist(), metrics, importance

model, feature_columns, metrics, importance = train_model()

st.subheader("Model Performance")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Accuracy", f"{metrics['Accuracy']*100:.2f}%")
c2.metric("Precision", f"{metrics['Precision']*100:.2f}%")
c3.metric("Recall", f"{metrics['Recall']*100:.2f}%")
c4.metric("F1-score", f"{metrics['F1-score']*100:.2f}%")
c5.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")

st.divider()

st.subheader("Enter Employee Details")

col1, col2 = st.columns(2)

with col1:
    satisfaction_level = st.slider(
        "Satisfaction Level",
        0.0, 1.0, 0.50, 0.01
    )

    last_evaluation = st.slider(
        "Last Evaluation Score",
        0.0, 1.0, 0.70, 0.01
    )

    number_project = st.number_input(
        "Number of Projects",
        min_value=1,
        max_value=15,
        value=4
    )

    average_monthly_hours = st.number_input(
        "Average Monthly Working Hours",
        min_value=50,
        max_value=400,
        value=200
    )

with col2:
    time_spend_company = st.number_input(
        "Years Spent in Company",
        min_value=1,
        max_value=20,
        value=3
    )

    work_accident = st.selectbox(
        "Work Accident",
        ["No", "Yes"]
    )

    promotion_last_5years = st.selectbox(
        "Promotion in Last 5 Years",
        ["No", "Yes"]
    )

    salary = st.selectbox(
        "Salary Bracket",
        ["low", "medium", "high"]
    )

if st.button("Predict Attrition", type="primary", use_container_width=True):

    row = {
        "satisfaction_level": satisfaction_level,
        "last_evaluation": last_evaluation,
        "number_project": number_project,
        "average_monthly_hours": average_monthly_hours,
        "time_spend_company": time_spend_company,
        "work_accident": 1 if work_accident == "Yes" else 0,
        "promotion_last_5years": 1 if promotion_last_5years == "Yes" else 0,
        "salary_low": salary == "low",
        "salary_medium": salary == "medium",
        "salary_high": salary == "high"
    }

    employee = pd.DataFrame([row])
    employee = employee.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(employee)[0]
    probability = model.predict_proba(employee)[0, 1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Employee is predicted to LEAVE (1)")
    else:
        st.success("Employee is predicted to STAY (0)")

    st.metric(
        "Probability of Leaving",
        f"{probability*100:.2f}%"
    )

st.divider()

st.subheader("Top Factors Driving Attrition")

chart_data = (
    importance
    .head(8)
    .set_index("Feature")[["Importance"]]
)

st.bar_chart(chart_data)

with st.expander("View Feature Importance Table"):
    st.dataframe(
        importance,
        use_container_width=True
    )
