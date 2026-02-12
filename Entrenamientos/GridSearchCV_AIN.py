import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import mlflow
import mlflow.sklearn
import functools

import os
os.makedirs("./plots", exist_ok=True)

def default_kwargs(**defaultKwargs):
    def actual_decorator(fn):
        @functools.wraps(fn)
        def g(*args, **kwargs):
            defaultKwargs.update(kwargs)
            merged = {**defaultKwargs, **kwargs}
            return fn(*args, **merged)
        return g
    return actual_decorator

#correr el experimento y enviar metricas a mlflow
def run_experiment(model_log, model, preprocessor, params, X_train, X_test, y_train, y_test, features_used):
    with mlflow.start_run(run_name=model_log["name"]):
        mlflow.set_tag("model", model_log["tag"])

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        gs = GridSearchCV(
            pipe,
            param_grid=params,
            cv=4,
            scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'],
            refit='neg_mean_squared_error',
            n_jobs=-1,
            return_train_score=True,
            verbose=2)

        gs.fit(X_train, y_train)

        try:
            best_model = gs.best_estimator_
        except ValueError:
            print("error en gridsearch.")
            return None

        #log params del best_model
        for param_name, param_value in gs.best_params_.items():
            mlflow.log_param(param_name, param_value)

        #---------- log metricas del mejor mod

        best_index = gs.best_index_

        best_cv_results = gs.cv_results_

        for metric in ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']:
            mlflow.log_metric(f"cv_mean_test_{metric}", best_cv_results[f"mean_test_{metric}"][best_index])
            mlflow.log_metric(f"cv_std_test_{metric}", best_cv_results[f"std_test_{metric}"][best_index])
            mlflow.log_metric(f"cv_mean_train_{metric}", best_cv_results[f"mean_train_{metric}"][best_index])
            mlflow.log_metric(f"cv_std_train_{metric}", best_cv_results[f"std_train_{metric}"][best_index])

        #guardar los resultados en un json
        mlflow.log_dict(gs.cv_results_, "cv_results.json")

        #---------- evaluar el best_model en test
        b_model = gs.best_estimator_
        y_pred = b_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("test_MAE", mae)
        mlflow.log_metric("test_MSE", mse)
        mlflow.log_metric("test_R2", r2)

        #---------- evaluar el best_model en train

        y_pred_t = b_model.predict(X_train)

        mae_t = mean_absolute_error(y_train, y_pred_t)
        mse_t = mean_squared_error(y_train, y_pred_t)
        r2_t = r2_score(y_train, y_pred_t)

        mlflow.log_metric("train_MAE", mae_t)
        mlflow.log_metric("train_MSE", mse_t)
        mlflow.log_metric("train_R2", r2_t)

        #---------- log true vs pred

        pred_df = pd.DataFrame({
            "original_index": X_test.index.to_list(),
            "y_true": y_test.values,
            "y_pred": y_pred
        })

        csv_str = pred_df.to_csv(index=False)
        mlflow.log_text(csv_str, f"{model_log['name']}_predictions.csv")

        #----------

        #log model_name y features usadas
        mlflow.log_param("model", model.__class__.__name__)
        mlflow.log_dict(features_used, "features_used.json")


        #------ guardar plot de pred vs true
        plot_path = f"./plots/{model_log['name']}_predictions_vs_real.png"
        sns.scatterplot(x = y_test, y = y_pred, alpha=0.5, s=5)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], c="red", linestyle="--")
        plt.title(f"R2: {r2} | MSE: {mse} | MAE: {mae} ")
        plt.xlabel("y_true")
        plt.ylabel("y_pred")

        plt.savefig(plot_path)
        plt.close()

        #log el plot
        mlflow.log_artifact(plot_path)

        mlflow.sklearn.log_model(b_model, model_log["name"])

        return [b_model, y_pred]

#Valores por defecto en caso de no entregar nada

models_default = {
    "MLPRegressor": {"tag": "", "model_conf": {"early_stopping": True, "validation_fraction":0.2}},
}

grid_default =  {
    'regressor__hidden_layer_sizes': [
        (10, 10),
        ],
    'regressor__activation': ['relu'],
    'regressor__alpha': [0.0001],
    'regressor__max_iter': [200]
}

cols_default = {
    "cols": [
        "NRO_PARADAS",
        "KM",
        "AREA_TOTAL_SUM",
        "PESO_TOTAL_SUM",
        "PRECIO_TOTAL_SUM",
        "EFICIENCIA_AREA",
        "DIA_NUM",
        "FRECUENCIA_PTE",
    ],
    "mean_cols": [
        "TRANSPORTE",
        "DESTINO",
    ],
    "cat_cols": []
}

# busca el modelo seleccionado segun el nombre y devuelve el objeto de modelo con los parametros iniciales
def load_model(model_name, paramss):
    match model_name:
        case "MLPRegressor":
            return MLPRegressor(**paramss)
        case "LinearR-poly":
            return Pipeline([
                        ("poly", PolynomialFeatures()),
                        ("lin", LinearRegression())
                    ])
        case "K-Neighbors Regressor":
            return KNeighborsRegressor(**paramss)
        case "RandomForestRegressor":
            return RandomForestRegressor(**paramss)
        case "HistGradientBoostingRegressor":
            return HistGradientBoostingRegressor(**paramss)
        case _:
            return None

#funcion principal que orquesta todo
@default_kwargs(mlf_url="http://localhost:5000", experiment="HPC-Pruebas", data_path="dataset_final.csv", models=models_default, params_grid=grid_default, cols_group=cols_default)
def gridsearch_run(**kwargs):
    
    mlflow_url = kwargs['mlf_url']
    mlflow_experiment = kwargs['experiment']
    dataset_path = kwargs['data_path']
    models = kwargs['models']
    param_grid = kwargs['params_grid']
    cols_group = kwargs['cols_group']


    mlflow.set_tracking_uri(mlflow_url)
    mlflow.set_experiment(mlflow_experiment)

    df = pd.read_csv(dataset_path, sep=';', header=0)

    #-------------- Transformación de datos + Pipelines

    pca_cols = cols_group['pca_cols']

    #para procesar columnas numericas
    scaler_cols = cols_group['cols']

    mean_cols = cols_group['mean_cols']

    #para procesar columnas categoricas (si aplica) -> one hot encoding
    cat_cols = cols_group['cat_cols']

    scaler_pipeline = Pipeline([
        ('scaler', RobustScaler())
    ])

    pca_pip = Pipeline([
        ('scaler', RobustScaler()),
        ('pca', PCA(1))
    ])

    te = Pipeline([
        ("target_enc", TargetEncoder(target_type="continuous", cv=3, smooth="auto", random_state=42)),
        ('scaler_after_te', RobustScaler())
    ])

    ohe_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('pca', pca_pip, pca_cols),
        ('num', scaler_pipeline, scaler_cols),
        ('mean', te, mean_cols),
        ('cat', ohe_pipeline, cat_cols)
    ])

    #-------------- separacion de los datos

    # split data
    y = df["COSTO"].astype(float)
    X = df[pca_cols + scaler_cols + mean_cols + cat_cols]  # según ajuste de columnas
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #-------------- entrenamientos

    # entrenamientos
    for name, model_ in models.items():
        model_log = {"tag": model_["tag"], "name": model_["name_log"]}
        model = load_model(name, model_["model_conf"])
        if model is not None:
            res = run_experiment(
                model_log=model_log,
                model=model,
                preprocessor=preprocessor,
                params = param_grid,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                features_used=cols_group
            )
        else:
            raise TypeError("No se encuentra un modelo válido")
