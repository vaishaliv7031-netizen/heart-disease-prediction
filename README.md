# Heart Disease Prediction Using Logistic Regression

## Project Overview

This project implements a machine learning classification model using **Logistic Regression** to analyze heart disease dataset records. It uses resting blood pressure (`trestbps`) and cholesterol (`chol`) as input features to predict the target class.

The project demonstrates model training, prediction, performance evaluation, and visualization using Python and popular machine learning libraries.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset

The project uses a CSV dataset named `dataset.csv`.

The following columns are used:

| Column | Description |
|---|---|
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `target` | Target class for classification |

The model uses `trestbps` and `chol` as independent variables (`X`) and `target` as the dependent variable (`y`).

## Project Workflow

1. **Data Loading:** Load the dataset using Pandas.
2. **Feature Selection:** Select resting blood pressure and cholesterol as input features.
3. **Model Training:** Train a Logistic Regression classifier using Scikit-learn.
4. **Prediction:** Generate predicted classes and positive-class probabilities.
5. **Performance Evaluation:** Calculate accuracy, precision, recall, and F1 score.
6. **Confusion Matrix:** Visualize classification results using a Seaborn heatmap.
7. **ROC Curve:** Plot the Receiver Operating Characteristic curve to examine the model's classification performance.

## Model Evaluation Metrics

- **Accuracy:** Measures the proportion of correctly classified records.
- **Precision:** Measures how many positive predictions are correct.
- **Recall:** Measures how many actual positive cases are correctly identified.
- **F1 Score:** Combines precision and recall into a single metric.
- **ROC Curve:** Illustrates the relationship between the true positive rate and false positive rate at different classification thresholds.

## Installation

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

1. Download or clone this repository.
2. Place `dataset.csv` in your project directory.
3. Update the CSV file path in the Python script if necessary.
4. Run the script:

```bash
python heart_disease_prediction.py
```

The program prints the evaluation metrics and displays the confusion matrix and ROC curve.

## Important Note

The current implementation trains and evaluates the model on the same dataset. Therefore, the reported metrics may be overly optimistic and may not represent performance on unseen data.

For a more reliable evaluation, use `train_test_split()` to separate training and testing data before fitting the model.

This project is intended for educational purposes only and is not a medically validated diagnostic tool.

## Future Improvements

- Implement a train-test split.
- Apply feature scaling.
- Evaluate the model using cross-validation.
- Compare Logistic Regression with other classification algorithms.
- Add hyperparameter tuning and threshold analysis.

## Author

Your Name

GitHub: Your GitHub Profile URL
