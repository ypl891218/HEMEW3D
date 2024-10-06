import os, sys
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
from mpi4py import MPI
from torch.distributed import init_process_group, destroy_process_group

import parse_HEME3D_arg, train_utils, ddp_trainer

def ddp_setup(rank, world_size, gpu_id, master_hostname):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = master_hostname
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(gpu_id)

if __name__ == '__main__':
    options = parse_HEME3D_arg.get_parsed_options()
    master_hostname = options['master_hostname']
    print("Master hostname: %s"%(master_hostname))

    world_size = MPI.COMM_WORLD.Get_size()
    world_rank = MPI.COMM_WORLD.Get_rank()
    gpu_id = world_rank % torch.cuda.device_count()
    ddp_setup(world_rank, world_size, gpu_id, master_hostname)

    print("World Rank: %s, World Size: %s, GPU_ID: %s"%(world_rank, world_size, gpu_id))
    print("Starting process on " + MPI.Get_processor_name() + ":" + torch.cuda.get_device_name(gpu_id))

    # name_config = f"FFNO3D-dv{dv}-{options['nlayers']}layers-S{S_in}-T{T_out}-padding{padding}-learningrate{str(learning_rate).replace('.','p')}-" \
    #     f"L1loss{str(loss_weights[0]).replace('.','p')}-L2loss{str(loss_weights[1]).replace('.','p')}-"
    # name_config += f"Ntrain{Ntrain}-batchsize{batch_size}"
    # name_config += options['additional_name']

    model = train_utils.get_model(options)
    name_config = train_utils.get_name_config(options)
    train_set = train_utils.get_dataset(options, "train")
    val_set = train_utils.get_dataset(options, "val")
    train_dataloader = train_utils.get_dataloader(train_set, options['batch_size'])
    val_dataloader = train_utils.get_dataloader(val_set, options['batch_size'])

    trainer = ddp_trainer.DDPTrainer(model, train_dataloader, val_dataloader, world_rank, gpu_id, options)
    trainer.train(name_config)
    destroy_process_group()