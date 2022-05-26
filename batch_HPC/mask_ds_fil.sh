#!/bin/bash
#SBATCH -p EM
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 30
#SBATCH -A phy210030p
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=akshay2
#SBATCH --output=/ocean/projects/phy210030p/akshay2/Slurm_logs/filmask_slurm_%j.log

# 42.67 GB memory per core for Bridges-LM partition.
# Ensure that the output directory to SBATCH exists prior to batch script execution.

# Define environment variables. $PROJECT = /ocean/projects/<group id>/<username>
SINGULARITY_CONT=$PROJECT/psrsearch.sif
CMDDIR=$PROJECT/HPC_pulsar/cmd_files

# Run acceleration searches within singularity container.
singularity exec -B /local $SINGULARITY_CONT $CMDDIR/mask_ds_fil.cmd
