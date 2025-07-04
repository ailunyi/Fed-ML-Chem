#!/bin/bash

#SBATCH -A chm250024-gpu         # allocation name
#SBATCH --nodes=1             # Total # of nodes 
#SBATCH --ntasks-per-node=1   # Number of MPI ranks per node (one rank per GPU)
#SBATCH --gpus-per-node=1   # Number of GPUs per node
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00        # Total run time limit (hh:mm:ss)
#SBATCH -J fl_job               # Job name
#SBATCH -o output_logs/run.o%j        # Name of stdout output file
#SBATCH -e error_logs/run.e%j          # Name of stderr error file
#SBATCH -p gpu               # Queue (partition) name
#SBATCH --mail-user=yi161@purdue.edu
#SBATCH --mail-type=all       # Send email to above address at begin and end of

# Manage processing environment, load compilers, and applications.
module purge
module load modtree/gpu
module load cuda/11.4.2

# Python environment

source /home/x-danoruo/miniconda3/etc/profile.d/conda.sh

conda activate fedml-gpu

which python
python --version

module list

export CUDA_LAUNCH_BLOCKING=1

echo "# Running on GPU"

cd $SLURM_SUBMIT_DIR

echo "## Loading dependencies"

# bash ../run-gpu.sh

echo "## Running model"

# Launch GPU code

#python QNN_federated/quantum_federated_dna_example.py

python RSNA_centralized.py

#python cleanup_logs.py