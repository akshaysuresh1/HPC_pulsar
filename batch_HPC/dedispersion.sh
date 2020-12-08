#!/bin/bash
#SBATCH -p RM-small
#SBATCH -t 00:02:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH -A phy200034p
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=akshay2
#SBATCH --output=/pylon5/phy200034p/akshay2/Test_logs/prepsubband_slurm_%j.log

# Ensure that the output directory to SBATCH exists prior to batch script execution.

# Define environment variables. $SCRATCH = /pylon5/<group id>/<username>
SINGULARITY_CONT=$SCRATCH/psrsearch.sif
CMDDIR=$SCRATCH/HPC_pulsar/cmd_files

# Load required modules.
module load singularity

# Run dedispersion module within singularity container.
singularity exec -B /local $SINGULARITY_CONT $CMDDIR/dedispersion.cmd
