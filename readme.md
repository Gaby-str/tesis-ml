## Estimación de Costos de Flete en Transporte de Productos de Cartón Mediante Machine Learning

### Generación de artefacto para Producción
---
El código para generar el artefacto de regresión destinado a ser usado para un entorno de producción se encuentra en ./dev_GenerarModelo

### Crear imagen docker con FastAPI para puesta de servicio en producción
---
El código necesario se encuentra en ./prod_ServicioPred
Se construye una imagen Docker a partir del código presente, el cual compone el servicio de API utilizando el artefacto .pkl para generar estimaciones.

### Imagen Docker
---
La **imagen Docker** es __frankrd1213/docker-model-service__ Esta entrega el servicio de predicciones mediante un servicio HTTP escuchando en el puerto 8000, con los siguientes endpoints :

* **/single_predict**: Para bajo volumen de solicitudes.
* **/csv_predict**: Para un alto volumen de solicitudes. Recibe y entrega archivos csv con los datos.
* **/reload_model_file**: Para reemplazar el actual modelo por uno actualizado, recibe modelos empaquetados en formato .pkl.

### Experimentos
---

Los archivos utilizados para realizar los experimentos se encuentran en la carpeta __Experimentos__, en donde las líneas de ejecución de trabajos de __Slurm__ están en __init_script.sh__.

__init_script.sh__ ejecuta un __Trabajo de Slurm__ con __script.sh__ por cada configuración (línea) especificada.

Caja trabajo comienza con el parser __run.py__, quien lee, corrobora y entrega los atributos del experimento al executor principal (__GridSearchCV_AIN.py__).

__GridSearchCV_AIN.py__ realiza todo el trabajo de procesamiento y GSCV del modelo seleccionado, generando métricas y resultados.