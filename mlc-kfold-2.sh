#!/bin/bash
#SBATCH --partition             fhs-fast
#SBATCH --ntasks                1
#SBATCH --nodes                 1
#SBATCH --tasks-per-node        1
#SBATCH --cpus-per-task         48
#SBATCH --mem-per-cpu           4G
#SBATCH --time                  120:00:00
#SBATCH --job-name              MLC-kfold
#SBATCH --output                log/MLC-kfold.%j.out
#SBATCH --error                 log/MLC-kfold.%j.err
#SBATCH --mail-type             ALL
##SBATCH --mail-user            chonloklei@um.edu.mo

source /etc/profile
source /etc/profile.d/modules.sh
source /home/chonloklei/m  # Load miniconda

ulimit -s unlimited

# Load module
module purge

# Path and Python version checks
pwd
python --version
conda activate /home/chonloklei/variant-classifier-md/env  # Load miniconda venv
python --version
which python

# Set up

# We are using multiprocessing, so switch multi-threading off
# https://stackoverflow.com/a/43897781
# export OMP_NUM_THREADS=1

# Run

python -u mlc-kfold-2.py --seed 0 --split variants
# python -u mlc-kfold-2.py --seed 1 --split frames

echo "Done."
