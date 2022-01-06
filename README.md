# HPC_pulsar
Pipeline with scripts for running pulsar search software (PRESTO and custom Python routines) on HPC nodes.

## Dependencies
```HPC_pulsar``` is written in Python 3, tested on Python 3.7.4 and has the following package dependencies.
- <a href="https://github.com/akshaysuresh1/psrdynspec">```psrdynspec```</a>
- <a href="https://github.com/scottransom/presto">```PRESTO```</a>

The above packages have several underlying dependencies that are listed in their respective repositories.

Note: To avoid the hassle of installing ```PRESTO``` and its several dependencies, you can pull a Singularity container encasing a full ```PRESTO``` installation <a href="https://singularity-hub.org/collections/4510">here </a>.

To make your life even simpler, <a href="https://singularity-hub.org/collections/4541">here</a> is another Singularity container with both ```PRESTO``` and ```psrdynspec``` installed. 

Batch scripts found the ```batch_HPC``` directory provide example command line calls run within a singularity container.

## Organization
1. ```batch_HPC```: Batch scripts for non-interactive execution via Slurm.
2. ```cmd_files```: Command line calls invoked by batch scripts for non-interactive program run.
3. ```config```: Configuration script of inputs to different modules.
4. ```executables```: Executable scripts that perform the computation. NOTE: Do not tamper with these scripts.

## Installation
1. Clone this repository to your local machine.
2. Edit relevant files in the ```batch_HPC```, ```cmd_files``` and ```config``` folders.
3. Pass relevant batch script to Slurm. You can now wait and relax while your code execution progresses non-interactively.

## Example
Say that you want to dedisperse some data using MPI-enabled ```prepsubband``` of <a href="https://github.com/scottransom/presto">```PRESTO```</a>.
1. Edit inputs in the configuration script ```config/dedisp.cfg```.
2. Verify that the paths to the relavant executable file and config script are correctly provided in the Shell script ```cmd_files/dedispersion.cmd```.
3. Supply appropriate parameters to Slurm in ```batch_HPC/dedispersion.sh```.
4. Type ```sbatch batch_HPC/dedispersion.sh``` and hit Enter.

## Troubleshooting
Please submit an issue to voice any problems or requests.

Improvements to the code are always welcome.
