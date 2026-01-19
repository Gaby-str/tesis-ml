from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Response
from contextlib import asynccontextmanager
from ModelHandler import ModelH
import pandas as pd
import json
import io
import aiofiles
import time

@asynccontextmanager
async def lifespan(app: FastAPI):
    # iniciar el objeto persistente
    app.state.ml_processor = ModelH("Model/model_pipeline.pkl")
    print("Aplicación iniciada, modelo cargado.")
    yield
    print("Aplicación cerrada")

app = FastAPI(lifespan=lifespan)

@app.get("/hello")
async def read_root():
    return {"message": "Hello World"}

@app.post("/single_predict")
async def single_predict(
    request: Request
):
    #obtener modelo
    processor = app.state.ml_processor
    if not processor.check_model_loaded():
        raise HTTPException(500, detail="Modelo no cargado correctamente en el servidor.")

    imput_preprocess_start = time.time()

    #transformar request en json
    input_data = json.dumps((await request.json()), indent=4, default=str)

    #transformar json a dataframe
    input_df = pd.read_json(input_data)

    #verificar que las features del json de entrada son las mismas que las del modelo
    if not list(input_df.columns) == await processor.get_features():
        raise HTTPException(status_code=400, detail="Las Columnas de entrada no coinciden con las que el modelo espera.")
    
    imput_preprocess_time = round((time.time() - imput_preprocess_start)*1000, 1)


    prediction_start = time.time()
    #realizar la prediccion
    preds = await processor.predict(input_df)
    prediction_time = round((time.time() - prediction_start)*1000, 1)
    
    return {"predictions": preds, "imput_preprocess_time": imput_preprocess_time, "prediction_time": prediction_time}

@app.post("/csv_predict")
async def csv_predict(file: UploadFile = File(...)):
    #asegurar el tipo de archivo
    if file.content_type != "text/csv":
        raise HTTPException(400, detail="Tipo de archivo no soportado. Se requiere un archivo CSV.")
    
    #obtener modelo
    processor = app.state.ml_processor
    if not processor.check_model_loaded():
        raise HTTPException(500, detail="Modelo no cargado correctamente en el servidor.")

    input_preprocess_start = time.time()
    #leer el archivo subido
    file_data = file.file

    #rransformar el archivo en dataframe
    try:
        input_df = pd.read_csv(file_data, sep=';', header=0)
    except Exception as e:
        raise HTTPException(400, detail=f"Error al leer el archivo CSV: {str(e)}")
    
    #verificar que las features del json de entrada son las mismas que las del modelo
    if not list(input_df.columns) == await processor.get_features():
        raise HTTPException(status_code=400, detail="Las Columnas de entrada no coinciden con las que el modelo espera.")
    
    input_preprocess_time = round((time.time() - input_preprocess_start)*1000, 1)

    prediction_start = time.time()
    #realizar la prediccion
    preds = await processor.predict(input_df)
    prediction_time = round((time.time() - prediction_start)*1000, 1)
    

    request_preprocess_start = time.time()
    #generar un archivo en memoria
    s_buf = io.StringIO()

    preds_df = pd.DataFrame(preds, columns=["PREDICTION"])

    preds_df.to_csv(s_buf, sep=';', index=False)

    request_preprocess_time = round((time.time() - request_preprocess_start)*1000, 1)


    print(f"Input Preprocess Time: {input_preprocess_time} ms")
    print(f"Prediction Time: {prediction_time} ms")
    print(f"Request Preprocess Time: {request_preprocess_time} ms")

    return Response(content=s_buf.getvalue(), headers={'Content-Disposition': 'attachment; filename="predictions.csv"'}, media_type="text/csv")

@app.post("/reload_model_file")
async def reload_model_file(file: UploadFile = File(...)):
    #obtener modelo
    processor = app.state.ml_processor
    
    #asegurar el tipo de archivo
    print(file.content_type)
    if file.content_type != "application/octet-stream":
        raise HTTPException(400, detail="Tipo de archivo no soportado. Se requiere un archivo .pkl o similar.")
    
    async with aiofiles.open("./Model/model_pipeline.pkl", 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    await processor.reload_model()

    return {"Result": "OK"}