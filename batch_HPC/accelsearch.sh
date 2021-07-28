#!/bin/bash
#SBATCH -p RM-shared
#SBATCH -t 28:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 9
#SBATCH -A phy200034p
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=akshay2
#SBATCH --output=/ocean/projects/phy200034p/akshay2/Slurm_logs/accelsearch_slurm_%j.log

# Ensure that the output directory to SBATCH exists prior to batch script execution.

# Define environment variables. $PROJECT = /ocean/projects/<group id>/<username>
SINGULARITY_CONT=$PROJECT/psrsearch.sif
CMDDIR=$PROJECT/HPC_pulsar/cmd_files

# Run acceleration searches within singularity container.
singularity exec -B /local $SINGULARITY_CONT $CMDDIR/accelsearch.cmd
