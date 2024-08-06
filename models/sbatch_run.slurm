#!/bin/bash
#SBATCH -J EAR23006        # Job name
#SBATCH -o EAR23006.o%j    # Name of stdout output file
#SBATCH -p gpu-a100        # Queue (partition) name
#SBATCH -N 1        # Total # of nodes
#SBATCH -n 1        # Total # of mpi tasks
#SBATCH -t 01:30:00        # Run time (hh:mm:ss)
#SBATCH -A EAR23006        # Allocation name (req'd if you have more than 1)

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

ibrun ./run_torchrun.sh 1 ${head_node_ip} 1
