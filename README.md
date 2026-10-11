# Heart Disease Prediction — Logistic Regression

A small Streamlit app that trains a Logistic Regression classifier using two features:
- `trestbps` — resting blood pressure
- `chol` — cholesterol
- `target` — binary target (0/1)

The app displays test-set accuracy, precision, recall, F1 score, a confusion matrix, an ROC curve, and an interactive sample prediction.

> **Important:** Educational project only. It is not a medical diagnostic tool.

## 1. Dataset format

Keep your dataset locally. It must be a CSV with these columns:

```csv
trestbps,chol,target
120,200,0
130,250,1
```

The values above are only illustrative examples, not real clinical guidance. Upload your actual `dataset.csv` through the app.

## 2. Run locally

Python 3.10+ recommended.

```bash
python -m venv .venv
```

Activate the environment:

- Linux/macOS: `source .venv/bin/activate`
- Windows: `.venv\\Scripts\\activate`

Install packages and launch:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## 3. Upload to GitHub

1. Create a new **public** GitHub repository, for example `heart-disease-logistic-regression`.
2. Extract this project ZIP.
3. Upload `app.py`, `requirements.txt`, `.gitignore`, and `README.md` to the repository root.
4. Do **not** upload private or sensitive data. Keep `dataset.csv` out of Git unless you have the right to share it.

Or use Git from this folder:

```bash
git init
git add app.py requirements.txt .gitignore README.md
git commit -m "Add heart disease logistic regression Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/heart-disease-logistic-regression.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. If Git says the remote already exists, use `git remote set-url origin <YOUR-REPOSITORY-URL>`.

## 4. Deploy with Streamlit Community Cloud

1. Push the project to GitHub.
2. Sign in at https://share.streamlit.io/ using GitHub.
3. Choose **Create app** and select your repository, branch (`main`), and main file path (`app.py`).
4. Click **Deploy**.
5. Open the deployed app and upload your CSV.

## 5. What changed from the original script?

- Uses a train/test split rather than evaluating on the same rows used to train the model.
- Standardizes the two numeric features in a pipeline.
- Handles missing/non-numeric rows and checks required columns.
- Calculates metrics on held-out test data.
- Uses a Streamlit interface for upload, metrics, charts, and sample predictions.

## Limitations / next improvements

- Only two features are used, so this is a learning baseline, not a clinically validated model.
- A single split can produce variable results; consider cross-validation.
- Check target meaning and label encoding in your particular dataset before interpreting predictions.
- Add model comparison, threshold analysis, and a reproducible evaluation report as later improvements.
