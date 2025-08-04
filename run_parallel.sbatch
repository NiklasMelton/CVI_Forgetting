#!/bin/bash
#SBATCH --job-name=OI_OFI_exp
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1                               #no. of cpu's per task
#SBATCH --mem-per-cpu=10G
#SBATCH --time=24:00:00
#SBATCH --out=Mill-%j.out
#SBATCH --mail-type=begin,end,fail,requeue
#SBATCH --mail-user=$nmmz76@umsystem.edu

# load python and activate env
module load python/3.12.1
source venv/bin/activate

# go to submission directory
cd $SLURM_SUBMIT_DIR

# run your main script
python main_parallel.py
