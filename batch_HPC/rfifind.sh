#!/bin/bash
#SBATCH -p RM-small
#SBATCH -t 00:05:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH -A phy200034p
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=akshay2
#SBATCH --output=/pylon5/phy200034p/akshay2/Test_logs/rfifind_slurm_%j.log

# Ensure that the output directory to SBATCH exists prior to batch script execution.

# Define environment variables. $SCRATCH = /pylon5/<group id>/<username>
SINGULARITY_CONT=$SCRATCH/psrsearch.sif
EXECDIR=$SCRATCH/HPC_pulsar/executables
CFGDIR=$SCRATCH/HPC_pulsar/config

# Load required modules.
module load singularity

# Run rfifind command within singularity container.
singularity exec -B /local $SINGULARITY_CONT python $EXECDIR/rfifind.py -i $CFGDIR/rfifind.cfg
