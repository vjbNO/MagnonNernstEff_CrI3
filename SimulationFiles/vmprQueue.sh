#!/bin/bash
#SBATCH --account=pet-vampire-2019
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=20
#SBATCH --time=10:00:00
#SBATCH --job-name=D_IP_Tr

module load OpenMPI/4.1.6-GCC-13.2.0

srun ./vampire-parallel
#mpirun --bind-to core --map-by core -np 20 ./vampire-parallel

