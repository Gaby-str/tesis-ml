import pandas as pd

from sklearn.preprocessing import RobustScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import HistGradientBoostingRegressor


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

# ------- config -------

random_state = 42

dataset_path = './dataset_final.csv'

scaler_cols = [
    "NRO_PARADAS",
    "KM",
    "AREA_TOTAL_SUM",
    "PESO_TOTAL_SUM",
    "PRECIO_TOTAL_SUM"
  ]

mean_cols = [
    "TRANSPORTE",
    "DESTINO",
    "TIPO"
  ]

target_col = "COSTO"

paramss = {
    "learning_rate": 0.08,
    "max_depth": 7,
    "max_leaf_nodes": 20,
    "min_samples_leaf": 13,
}

test_size = 0.2

# ------- Leer dataset ------- 

df = pd.read_csv(dataset_path, sep=';', header=0)

# ------- Pipeline ------- 

model = HistGradientBoostingRegressor(**paramss)

scaler_pipeline = Pipeline([
    ('scaler', RobustScaler())
])


te = Pipeline([
    ("target_enc", TargetEncoder(target_type="continuous", cv=3, smooth="auto", random_state=random_state)),
    ('scaler_after_te', RobustScaler())
])

preprocessor = ColumnTransformer([
    ('num', scaler_pipeline, scaler_cols),
    ('mean', te, mean_cols)
])

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# ------- Split Data ------- 

# split data
y = df[target_col].astype(float)
X = df[scaler_cols + mean_cols] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)





# ------- Train/Pred Pipe ------- 

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# ------- Show Metrics ------- 

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R2: {r2}")

# ------- Generar PKL -------

with open('model_pipeline.pkl', 'wb') as f:
    pickle.dump(pipe, f)
    print("Modelo guardado como model_pipeline.pkl")