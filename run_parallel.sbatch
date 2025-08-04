#!/bin/bash
#SBATCH --job-name=OI_OFI_exp
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=10000
#SBATCH --time=24:00:00
#SBATCH --out=Mill-%j.out
#SBATCH --mail-type=begin,end,fail,requeue

# load python and activate env
module load python/3.12.1
source venv/bin/activate

# go to submission directory
cd $SLURM_SUBMIT_DIR

# run your main script
python main_parallel.py
