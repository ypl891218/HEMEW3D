# Create virtural enviroment and install dependencies
```
python3 -m venv /path/to/virtual-env  # (e.g., $SCRATCH/python-envs/test)
source /path/to/virtual-env/bin/activate
pip install -r requirements.txt
```

# Data preprocessorsing
1. Copy material and velocity file to ```HEMEW3D/data/``` with names like ```materials0-1999.npy``` and ```sample1.h5```
2. Run ```HEMEW3D/models/preprocessing/create_data_velocityfields_v2.py``` with desired arguments
```
@Ntrain, default=27000, help="Number of training samples"
@Nval, default=3000, help="Number of test samples"
@S_out, default=32, help="Size of the sensors array (interpolated)"
@Nt, default=320, help="Number of time steps"
@f, default=50, help="Sampling frequency"
@fmax, default=5, help="Maximum frequency to filter signals"
@Irun0, default=100000, help='Index of the first element to use in the folder path_u'
```

# Train the model
1. Go to ```HEMEW3D/models/``` edit ```sbatch-run.slurm``` with desired config
- example:
```
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
```
2. Edit ```run_torchrun.sh`` with desired config
- example
```
#!/bin/bash
# run_torchrun.sh

. /scratch/10049/weichusheng/python-envs/test/bin/activate
torchrun \
    --nnodes=$1 \
    --nproc_per_node=$3 \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$2:29500 \
    /scratch/10049/weichusheng/scratchBackup/HEMEW3D/models/train_ffno3d_DDP_torchrun_multinode.py
```
```
--nnodes: number of nodes
--nproc_per_node: number of gpus per node
--rdzv_endpoint: headnode:port
```

