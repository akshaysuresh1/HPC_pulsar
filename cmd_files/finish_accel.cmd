mpirun -n $SLURM_NTASKS \
        singularity exec -B /local /ocean/projects/phy210030p/akshay2/psrsearch.sif \
        python /ocean/projects/phy210030p/akshay2/HPC_pulsar/executables/finish_accel.py \
       -i /ocean/projects/phy210030p/akshay2/HPC_pulsar/config/finish_accel.cfg
