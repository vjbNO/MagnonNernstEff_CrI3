#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --account=pet-vampire-2019
#SBATCH --job-name=hashMuWithDM
#SBATCH --nodes=1
#SBATCH --ntasks=25

module load Python/3.11.3-GCCcore-12.3.0
module load scipy
module load matplotlib

python HashMu.py

