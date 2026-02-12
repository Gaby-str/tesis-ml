#!/usr/bin/env python3
"""
El config.json debe tener la forma:
{
  "script_path": "train.py",
  "function_name": "gridsearch_run",
  "kwargs": {
      "mlf_url": "...",
      "experiment": "...",
      "data_path": "...",
      "models": { ... },
      "params_grid": { ... },
      "cols_group": { ... }
  }
}
"""
import argparse
import json
import runpy
import os
import sys
import traceback

def main():
    parser = argparse.ArgumentParser(description="Wrapper para ejecutar gridsearch_run desde un script de python utilizando una configuracion JSON.")
    parser.add_argument('--config', '-c', required=True, help='Ruta para el archivo JSON de configuracion')
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Archivo de configuracion no encontrado: {config_path}", file=sys.stderr)
        sys.exit(2)

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    script_path = cfg.get('script_path', 'GridSearchCV_AIN.py')
    function_name = cfg.get('function_name', 'gridsearch_run')
    kwargs = cfg.get('kwargs', {})

    if not os.path.exists(script_path):
        print(f"Script de python no encontrado: {script_path}", file=sys.stderr)
        sys.exit(3)

    print(f"Corriendo Script: {script_path}")
    print(f"Buscando funcion: {function_name}")
    print(f"Con las kwargs keys: {list(kwargs.keys())}")

    try:
        # Ejecuta el script y captura su namespace
        module_globals = runpy.run_path(script_path, run_name="__main__")
    except Exception as e:
        print("Error al ejecutar script_path con runpy.run_path():", file=sys.stderr)
        traceback.print_exc()
        sys.exit(4)

    if function_name not in module_globals:
        print(f"Función '{function_name}' no encontrada en {script_path}. Objetos-funciones disponibles: {list(module_globals.keys())}", file=sys.stderr)
        sys.exit(5)

    fn = module_globals[function_name]
    if not callable(fn):
        print(f"Objeto '{function_name}' en {script_path} no es un invocable.", file=sys.stderr)
        sys.exit(6)

    try:
        print("Invocando función...")
        result = fn(**kwargs)
        print("Ejecución terminada. Resultado:", type(result).__name__ if result is not None else None)
    except Exception:
        print("Error al invocar la función:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(7)

if __name__ == "__main__":
    main()
