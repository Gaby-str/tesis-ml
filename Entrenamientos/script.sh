#!/bin/bash
#SBATCH --job-name=train
#SBATCH --cpus-per-task=14
#SBATCH --mem=90G
#SBATCH --error=generic.err
#SBATCH --output=generic.out
#SBATCH --nodelist=tokikura,host[13]

# Crear carpetas de logs si no existen
mkdir -p logs

echo "Job $SLURM_JOB_ID iniciado en $(hostname) a las $(date)"
echo "Directorio de trabajo: $(pwd)"

#--------- Entorno Python
micromamba activate MLFlow
python --version

#--------- Ejecutar el wrapper que leera config.json
CONFIG_PATH=$1
WRAPPER="run_.py"

echo "Usando configuración: $CONFIG_PATH"
python -u "${WRAPPER}" -c "${CONFIG_PATH}"

echo "Job terminado el $(date)"