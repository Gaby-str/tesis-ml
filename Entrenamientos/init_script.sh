#!/bin/bash

sbatch --job-name=train_MLP_all --nodelist=host[01] --error=./outs_new/train_MLP1_all.err --output=./outs_new/train_MLP1_all.out ./script.sh ./configs_new/configMLP1_all_f.json
sbatch --job-name=train_MLP2_all --nodelist=host[05] --error=./outs_new/train_MLP2_all.err --output=./outs_new/train_MLP2_all.out ./script.sh ./configs_new/configMLP2_all_f.json
sbatch --job-name=train_RF_all --nodelist=v100 --mem=200G --cpus-per-task=30 --error=./outs_new/train_RF_all.err --output=./outs_new/train_RF_all.out ./script.sh ./configs_new/configRF_all_f.json
sbatch --job-name=train_KNN_all --nodelist=host[13] --error=./outs_new/train_KNN_all.err --output=./outs_new/train_KNN_all.out ./script.sh ./configs_new/configKNN_all_f.json
sbatch --job-name=train_HGB_all --nodelist=host[32] --error=./outs_new/train_HGB_all.err --output=./outs_new/train_HGB_all.out ./script.sh ./configs_new/configHGB_all_f.json
sbatch --job-name=train_POLY_all --nodelist=host[34] --error=./outs_new/train_POLY_all.err --output=./outs_new/train_POLY_all.out ./script.sh ./configs_new/configPOLY_all_f.json

sbatch --job-name=train_MLP_reduced --nodelist=host[01] --error=./outs_new/train_MLP1_reduced.err --output=./outs_new/train_MLP1_reduced.out ./script.sh ./configs_new/configMLP1_reduced_f.json
sbatch --job-name=train_MLP2_reduced --nodelist=host[05] --error=./outs_new/train_MLP2_reduced.err --output=./outs_new/train_MLP2_reduced.out ./script.sh ./configs_new/configMLP2_reduced_f.json
sbatch --job-name=train_RF_reduced --nodelist=v100 --mem=200G --cpus-per-task=30 --error=./outs_new/train_RF_reduced.err --output=./outs_new/train_RF_reduced.out ./script.sh ./configs_new/configRF_reduced_f.json
sbatch --job-name=train_KNN_reduced --nodelist=host[13] --error=./outs_new/train_KNN_reduced.err --output=./outs_new/train_KNN_reduced.out ./script.sh ./configs_new/configKNN_reduced_f.json
sbatch --job-name=train_HGB_reduced --nodelist=host[32] --error=./outs_new/train_HGB_reduced2.err --output=./outs_new/train_HGB_reduced2.out ./script.sh ./configs_new/configHGB_reduced_f.json
sbatch --job-name=train_POLY_reduced --nodelist=host[34] --error=./outs_new/train_POLY_reduced.err --output=./outs_new/train_POLY_reduced.out ./script.sh ./configs_new/configPOLY_reduced_f.json

#sbatch --job-name=HGB_PCA --error=./outs/HGB_PCA.err --output=./outs/HGB_PCA.out ./script.sh ./configs/configHGB_PCA.json

#sbatch --job-name=test --nodelist=host[24] --error=./outs/p.err --output=./outs/p.out ./script.sh

sbatch --job-name=TN_GSCV --error=./outs_new/TN_GSCV1.err --output=./outs_new/TN_GSCV1.out ./TN_init_script_gs.sh ./configs/configTN.json

sbatch --job-name=train_HGB_all --nodelist=host[32] --error=./outs_new/train_HGB_all2.err --output=./outs_new/train_HGB_all2.out ./script.sh ./configs_new/configHGB_all_f_2.json
