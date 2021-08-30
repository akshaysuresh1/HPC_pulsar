mpirun -n $SLURM_NTASKS \
        singularity exec -B /local /ocean/projects/phy200034p/akshay2/psrsearch.sif \
        python /ocean/projects/phy200034p/akshay2/HPC_pulsar/executables/accelsearch_sift_fold.py \
       -i /ocean/projects/phy200034p/akshay2/HPC_pulsar/config/accel.cfg
