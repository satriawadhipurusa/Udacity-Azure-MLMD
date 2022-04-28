import argparse
import joblib
import logging
import json
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core.run import Run

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    level=logging.DEBUG)


def get_model(use_cols: list = []) -> sklearn.pipeline.Pipeline:
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler", MinMaxScaler())
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numerical_pipeline, use_cols),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("algo", SVC(max_iter=500, probability=True))
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=1.0,
                        help="Kernel coefficient one of (‘rbf’, ‘poly’, ‘sigmoid’) or `float`")
    parser.add_argument('--C', type=float, default=1.0,
                        help="Regularization parameter. Must be strictly positive. The penalty is a squared l2 penalty.")
    parser.add_argument('--class_weight', type=str, default=None,
                        help="Whether to balance the weight inversely proportional to the frequency of classes or not")
    args = parser.parse_args()
    args.class_weight = eval(args.class_weight)

    run = Run.get_context()

    # get data
    url = "https://media.githubusercontent.com/media/satriawadhipurusa/ml-dataset-collection/master/Fraud-Detection/creditcard-fraud.csv"
    logging.info(f"Downloading data from .....{url}")
    ds = TabularDatasetFactory.from_delimited_files(url)
    df = ds.to_pandas_dataframe()

    logging.info("Splitting data to train/test.....")
    X, y = df.drop(columns="Class"), df.Class
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=123)

    fit_args = {'algo__' + k: v for k, v in vars(args).items()}
    model = get_model(use_cols=X.columns.tolist())
    logging.info(f"Fitting model..... {model}")
    model.set_params(**fit_args)
    model.fit(X_train, y_train)

    y_scores = cross_val_predict(model, X, y, cv=3, method="predict_proba")
    auc_weighted = roc_auc_score(y, y_scores[:, 1], average="weighted")
    logging.info(f"Finished with AUC Score of {auc_weighted}")
    run.log("AUC_Weighted", auc_weighted)

    os.makedirs('outputs', exist_ok=True)
    namefile = 'outputs/model.joblib'
    logging.info(f"Saving model into .....{namefile}")
    joblib.dump(model, namefile)
