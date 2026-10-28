import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

st.set_page_config(
    page_title="Predicting Bestseller Books",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Predicting Bestseller Books")
st.caption("Machine-learning dashboard for classifying high-selling books")

st.info(
    "This dataset does not contain an official bestseller label. "
    "The dashboard defines High-Selling as the top 25% of books by units sold. "
    "This is a proxy label, not a verified industry bestseller designation."
)

@st.cache_data
def load_default_data():
    for filename in ["cleaneddataset.csv", "Books_Data_Clean.csv"]:
        try:
            data = pd.read_csv(filename)
            data.columns = data.columns.str.strip()
            return data, filename
        except FileNotFoundError:
            continue
    return None, None

uploaded_file = st.sidebar.file_uploader("Upload a CSV dataset (optional)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    source_name = uploaded_file.name
else:
    df, source_name = load_default_data()

if df is None:
    st.warning("Dataset not found. Upload your CSV using the sidebar, or place cleaneddataset.csv beside app.py.")
    st.stop()

df.columns = df.columns.str.strip()
df = df.drop(columns=["index"], errors="ignore")

required_column = "units sold"
if required_column not in df.columns:
    st.error("The dataset must contain a column named 'units sold'. Check the column name and upload the correct CSV.")
    st.stop()

df["units sold"] = pd.to_numeric(df["units sold"], errors="coerce")
df = df.dropna(subset=["units sold"]).copy()
if df.empty:
    st.error("No valid numeric values were found in the 'units sold' column.")
    st.stop()

threshold = df["units sold"].quantile(0.75)
df["High_Selling"] = (df["units sold"] >= threshold).astype(int)

excluded = [
    "High_Selling", "units sold", "sales rank",
    "gross sales", "publisher revenue", "Book Name"
]
X_raw = df.drop(columns=excluded, errors="ignore").copy()
y = df["High_Selling"].copy()

if X_raw.shape[1] == 0:
    st.error("No usable input features remain after excluding target-leakage columns.")
    st.stop()

numeric_cols = X_raw.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

for col in numeric_cols:
    X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")
    median_value = X_raw[col].median()
    X_raw[col] = X_raw[col].fillna(median_value if pd.notna(median_value) else 0)

for col in categorical_cols:
    mode_values = X_raw[col].mode(dropna=True)
    fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
    X_raw[col] = X_raw[col].fillna(fill_value).astype(str)

X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True, dtype=int)
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

if y.nunique() < 2 or y.value_counts().min() < 2:
    st.error("The target needs at least two examples in each class to create a stratified train/test split.")
    st.stop()

test_size = st.sidebar.slider("Test data (%)", min_value=15, max_value=35, value=20, step=5) / 100
random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

# Scale only numeric columns; one-hot encoded columns remain unchanged.
scaler = StandardScaler()
scaled_numeric_cols = [c for c in numeric_cols if c in X_train.columns]
if scaled_numeric_cols:
    X_train.loc[:, scaled_numeric_cols] = scaler.fit_transform(X_train[scaled_numeric_cols])
    X_test.loc[:, scaled_numeric_cols] = scaler.transform(X_test[scaled_numeric_cols])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(random_state=random_state, class_weight="balanced", max_depth=8),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced")
}

trained_models = {}
results = []
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    })
    trained_models[model_name] = model

results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
best_name = results_df.iloc[0]["Model"]
selected_model_name = st.sidebar.selectbox("Model for evaluation and prediction", list(models.keys()), index=list(models.keys()).index(best_name))
selected_model = trained_models[selected_model_name]
test_predictions = selected_model.predict(X_test)
test_probabilities = selected_model.predict_proba(X_test)[:, 1]

tab_overview, tab_models, tab_predict, tab_data = st.tabs(
    ["Overview", "Model Performance", "Predict a Book", "Dataset Explorer"]
)

with tab_overview:
    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Books in dataset", f"{len(df):,}")
    c2.metric("Input features", f"{len(numeric_cols) + len(categorical_cols)}")
    c3.metric("High-selling threshold", f"{threshold:,.0f} units")
    c4.metric("Best F1 model", best_name)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### High-selling class distribution")
        counts = df["High_Selling"].map({0: "Not High-Selling", 1: "High-Selling"}).value_counts()
        st.bar_chart(counts)
    with c2:
        st.markdown("#### Units sold distribution")
        fig, ax = plt.subplots()
        ax.hist(df["units sold"], bins=20)
        ax.set_xlabel("Units sold")
        ax.set_ylabel("Number of books")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Selected input features")
    st.write(", ".join(X_raw.columns.tolist()))
    st.markdown("#### Excluded columns")
    st.write("`units sold`, `sales rank`, `gross sales`, `publisher revenue`, `Book Name`, and the row index (if present).")

with tab_models:
    st.subheader("Compare classification models")
    st.dataframe(results_df.style.format({
        "Accuracy": "{:.3f}", "Precision": "{:.3f}",
        "Recall": "{:.3f}", "F1 Score": "{:.3f}", "ROC-AUC": "{:.3f}"
    }), use_container_width=True)

    fig, ax = plt.subplots()
    results_chart = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]]
    results_chart.plot(kind="bar", ax=ax, ylim=(0, 1), rot=20)
    ax.set_ylabel("Score")
    ax.set_title("Model metric comparison")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(f"#### Classification report — {selected_model_name}")
    st.code(classification_report(
        y_test, test_predictions,
        labels=[0, 1],
        target_names=["Not High-Selling", "High-Selling"],
        zero_division=0
    ))

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_test, test_predictions,
        display_labels=["Not High-Selling", "High-Selling"],
        values_format="d", ax=ax
    )
    ax.set_title(f"Confusion matrix — {selected_model_name}")
    st.pyplot(fig)
    plt.close(fig)

with tab_predict:
    st.subheader("Try a book prediction")
    st.write("Enter the book details available in the dataset. Fields excluded from training are not requested.")
    with st.form("book_prediction_form"):
        entered = {}
        col1, col2 = st.columns(2)
        for i, col in enumerate(X_raw.columns):
            target_col = col1 if i % 2 == 0 else col2
            if col in numeric_cols:
                values = pd.to_numeric(X_raw[col], errors="coerce").dropna()
                default_value = float(values.median()) if not values.empty else 0.0
                min_value = float(values.min()) if not values.empty else 0.0
                max_value = float(values.max()) if not values.empty else max(default_value, 1.0)
                if min_value == max_value:
                    max_value = min_value + 1.0
                with target_col:
                    entered[col] = st.number_input(
                        col, min_value=min_value, max_value=max_value,
                        value=min(max(default_value, min_value), max_value),
                        key=f"input_{col}"
                    )
            else:
                options = sorted(X_raw[col].astype(str).unique().tolist())
                if not options:
                    options = ["Unknown"]
                with target_col:
                    entered[col] = st.selectbox(col, options, key=f"input_{col}")

        submitted = st.form_submit_button("Predict high-selling status")

    if submitted:
        new_row = pd.DataFrame([entered])
        for col in numeric_cols:
            if col in new_row.columns:
                new_row[col] = pd.to_numeric(new_row[col], errors="coerce").fillna(X_raw[col].median())
        for col in categorical_cols:
            if col in new_row.columns:
                new_row[col] = new_row[col].fillna("Unknown").astype(str)
        new_encoded = pd.get_dummies(new_row, columns=categorical_cols, drop_first=True, dtype=int)
        new_encoded = new_encoded.reindex(columns=X.columns, fill_value=0)
        if scaled_numeric_cols:
            new_encoded.loc[:, scaled_numeric_cols] = scaler.transform(new_encoded[scaled_numeric_cols])
        prediction = selected_model.predict(new_encoded)[0]
        probability = selected_model.predict_proba(new_encoded)[0, 1]
        if prediction == 1:
            st.success("Prediction: High-Selling")
        else:
            st.warning("Prediction: Not High-Selling")
        st.metric("Estimated probability of High-Selling", f"{probability:.1%}")
        st.caption("This is a model estimate for the dataset's top-quartile proxy, not a guarantee of future sales or an official bestseller status.")

with tab_data:
    st.subheader("Explore the dataset")
    search_text = st.text_input("Search rows (book, author, publisher, genre, etc.)")
    view_df = df.copy()
    if search_text:
        text_cols = view_df.select_dtypes(include=["object"]).columns
        if len(text_cols):
            mask = view_df[text_cols].astype(str).apply(
                lambda col: col.str.contains(search_text, case=False, na=False)
            ).any(axis=1)
            view_df = view_df[mask]
    st.write(f"Showing {len(view_df):,} rows")
    st.dataframe(view_df, use_container_width=True, height=420)
    st.download_button(
        "Download current table as CSV",
        data=view_df.to_csv(index=False).encode("utf-8"),
        file_name="books_dataset_view.csv",
        mime="text/csv"
    )

st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {source_name}")
st.sidebar.caption("Target: top 25% by units sold")
