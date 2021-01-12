#!/usr/bin/env bash
#SBATCH --output=/Midgard/home/%u/output/jobs/job%J.out
#SBATCH --error=/Midgard/home/%u/output/jobs/job%J.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=miquelmr@kth.se
#SBATCH --constrain="khazadum|rivendell|belegost|shire|gondor|balrog|shelob|smaug"
#SBATCH --mem=20GB
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=5
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
echo "Nodelist ${SLURM_NODELIST}"
srun nvidia-smi
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $1
echo "Using conda env: $CONDA_PREFIX"
echo "Current dir: ${PWD}"
export PYTHONPATH=.
nodename=${SLURMD_NODENAME}

srun python classification.py --dist_url $nodename --data_location $2
