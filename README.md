# Predicting Bestseller Books

A machine-learning project that explores book sales data and predicts whether a book belongs to the high-selling group. The project includes a Streamlit dashboard, dataset, and SI Lab notebook/report.

## Important note about the target

The source dataset does not contain an official bestseller label. `High_Selling` is a proxy label based on the top 25% of `units sold` in the dataset (the project's reported threshold is 5,312 units). It should not be interpreted as an official bestseller designation.

To reduce target leakage, fields such as `units sold`, `sales rank`, `gross sales`, and `publisher revenue` are not used as model input features.

## Files

- `app.py` — Streamlit web interface
- `cleaneddataset.csv` — cleaned book dataset used by the dashboard
- `B2_50_SI_LA2_Project_Probability_CLT_Hypothesis_Testing.ipynb` — notebook for analysis, CLT, hypothesis testing, and model evaluation
- `Predicting_Bestseller_Books_SI_Lab_Report.pdf` — project report
- `Predicting_Bestseller_Books_SI_Lab_Report_Editable.docx` — editable report
- `requirements.txt` — Python dependencies

## Run the dashboard locally

1. Install Python 3.10 or newer.
2. Download or clone this repository and open a terminal in the project folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Run the notebook

Open the `.ipynb` file in Google Colab or Jupyter. Keep `cleaneddataset.csv` in the same working folder, or upload it when prompted by your notebook workflow.

## Project workflow

1. Load and explore the cleaned dataset.
2. Analyze distributions and the target-class balance.
3. Demonstrate the Central Limit Theorem using repeated samples.
4. Conduct a chi-square test and Welch's t-test.
5. Train and compare Logistic Regression, Decision Tree, and Random Forest classifiers.
6. Use the dashboard to enter book attributes and view a prediction.

## Limitations

The model predicts a dataset-defined high-selling category, not guaranteed future sales or an official bestseller status. Performance depends on dataset quality, feature availability, and the chosen target definition.
