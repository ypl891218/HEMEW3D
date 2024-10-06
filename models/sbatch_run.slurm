#!/bin/bash
#SBATCH -J yunping        # Job name
#SBATCH -o yunping.o%j    # Name of stdout output file
#SBATCH -p gh         # Queue (partition) name
#SBATCH -N 4        # Total # of nodes
#SBATCH -n 4        # Total # of mpi tasks
#SBATCH -t 24:00:00        # Run time (hh:mm:ss)
#SBATCH -A EAR24026        # Allocation name (req'd if you have more than 1)

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Head Node: $head_node
echo Node IP: $head_node_ip
export LOGLEVEL=INFO

module load gcc cuda
module load python3_mpi
source $WORK/venv/bin/activate
module load python3_mpi

ibrun -np 4 ./script_ffno3d_multi_gpu.sh ${head_node}
