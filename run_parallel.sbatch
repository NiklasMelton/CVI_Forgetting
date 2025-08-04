#!/bin/bash
#SBATCH --job-name=OI_OFI_exp
#SBATCH --partition=general
#SBATCH --nodelist=compute-14-02
#SBATCH --nodes=1
#SBATCH --ntasks=20               # spawn 20 independent tasks
#SBATCH --cpus-per-task=2         # give each task 3 CPUs for its internal threading
#SBATCH --mem-per-cpu=4G          # 4 GB × 3 CPUs × 20 tasks = 240 GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/Mill-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=nmmz76@umsystem.edu

module load python/3.12.1
source venv/bin/activate

cd $SLURM_SUBMIT_DIR
python main_parallel.py
