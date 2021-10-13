#!/bin/bash
#SBATCH -p RM
#SBATCH -t 47:30:00
#SBATCH -N 3
#SBATCH --ntasks-per-node 128
#SBATCH -A phy210030p
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=akshay2
#SBATCH --output=/ocean/projects/phy210030p/akshay2/Slurm_logs/multi_acceljerk_%j.log

# Ensure that the output directory to SBATCH exists prior to batch script execution.
module load openmpi/3.1.6-gcc8.3.1

# Define environment variables. $PROJECT = /ocean/projects/<group id>/<username>
CMDDIR=$PROJECT/HPC_pulsar/cmd_files

echo $SLURM_NTASKS
# Run acceleration searches within singularity container.
$CMDDIR/multinode_accelsearch.cmd
