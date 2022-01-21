#!/bin/bash
#SBATCH -A m1727_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:01:00
#SBATCH -n 16
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=map_gpu:0,1,2,3

export SLURM_CPU_BIND="cores"
srun python demo.py --gpu_cluster_n_hosts=16 --gpu_cluster_host_id=$SLURM_PROCID