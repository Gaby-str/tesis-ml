FROM python:3.12.10

WORKDIR /code

COPY ./MLprod_requirements.txt /code/MLprod_requirements

RUN pip install --no-cache-dir --upgrade -r /code/MLprod_requirements

COPY ./Model/model_pipeline.pkl ./Model/model_pipeline.pkl
COPY ./main.py .
COPY ./ModelHandler.py .

CMD ["fastapi","run","./main.py","--port","8000"]
