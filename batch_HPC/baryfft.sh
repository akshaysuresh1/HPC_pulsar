#!/bin/bash
#SBATCH -p RM-shared
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 4
#SBATCH -A phy210030p
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=akshay2
#SBATCH --output=/ocean/projects/phy210030p/akshay2/Slurm_logs/baryfft_slurm_%j.log

# Ensure that the output directory to SBATCH exists prior to batch script execution.

# Define environment variables. $PROJECT = /ocean/projects/<group id>/<username>
SINGULARITY_CONT=$PROJECT/psrsearch.sif
CMDDIR=$PROJECT/HPC_pulsar/cmd_files

# Run barycenter + FFT module within singularity container.
singularity exec -B /local $SINGULARITY_CONT $CMDDIR/baryfft.cmd
