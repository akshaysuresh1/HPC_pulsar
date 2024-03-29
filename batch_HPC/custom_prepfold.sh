#!/bin/bash
#SBATCH -p RM-shared
#SBATCH -t 20:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH -A phy210030p
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=akshay2
#SBATCH --output=/ocean/projects/phy210030p/akshay2/Slurm_logs/customfold_slurm_%j.log

# Ensure that the output directory to SBATCH exists prior to batch script execution.

# Define environment variables. $PROJECT = /ocean/projects/<group id>/<username>
SINGULARITY_CONT=$PROJECT/psrsearch.sif
EXECDIR=$PROJECT/HPC_pulsar/executables
CFGDIR=$PROJECT/HPC_pulsar/config

# Run rfifind command within singularity container.
singularity exec -B /local $SINGULARITY_CONT python $EXECDIR/custom_prepfold.py -i $CFGDIR/custom_prepfold.cfg
