#!/bin/bash
#SBATCH -p LM
#SBATCH -t 08:00:00
#SBATCH --mem=192GB
#SBATCH -A phy200034p
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=akshay2
#SBATCH --output=/pylon5/phy200034p/akshay2/Slurm_logs/filmask_slurm_%j.log

# 48 GB memory per core for Bridges-LM partition.
# Ensure that the output directory to SBATCH exists prior to batch script execution.

# Define environment variables. $SCRATCH = /pylon5/<group id>/<username>
SINGULARITY_CONT=$SCRATCH/psrsearch.sif
CMDDIR=$SCRATCH/HPC_pulsar/cmd_files

# Load required modules.
module load singularity

# Run acceleration searches within singularity container.
singularity exec -B /local $SINGULARITY_CONT $CMDDIR/mask_ds_fil.cmd
