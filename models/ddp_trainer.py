import h5py
import numpy as np
import timeit
import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from models.functions.helpers_deeplearning import get_device, get_batch_size, loss_criterion, RunningAverage, EarlyStopper

class DDPTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        rank: int,
        gpu_id: int,
        options,
    ) -> None:
        self.rank = rank
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optim.Adam(model.parameters(), lr=options['learning_rate'], betas=(0.9, 0.999))
        self.val_data = val_data
        self.model = DDP(model, device_ids=[gpu_id])
        self.options = options
        self.epochs = options['epochs']
        self.loss_weights = options['loss_weights']
        self.patience = 60

    def train(self, name_config: str):
        # name_config = f"FFNO3D-dv{dv}-{self.nlayers}layers-S{S_in}-T{T_out}-padding{padding}-learningrate{str(learning_rate).replace('.','p')}-" \
        # f"L1loss{str(loss_weights[0]).replace('.','p')}-L2loss{str(loss_weights[1]).replace('.','p')}-"
        # name_config += f"Ntrain{Ntrain}-batchsize{batch_size}"
        # name_config += options['additional_name']

        # Store losses history
        train_history = {'loss_relative':[], 'loss_absolute':[]}
        val_history = {'loss_relative':[], 'loss_absolute':[]}
        best_loss = np.inf

        ### OPTIMIZER
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=10, verbose=True) 
        early_stopper = EarlyStopper(patience=self.patience, min_delta=0.0001)

        ### TRAINING       
        for ep in range(self.epochs):
            self.train_data.sampler.set_epoch(ep)
            t1 = timeit.default_timer()
            self.model.train()
            train_losses_relative = RunningAverage()
            train_losses_absolute = RunningAverage()
        
            # training
            for _ in self.train_data:
                a = _[0].to(self.gpu_id)
                uE = _[1].to(self.gpu_id)
                uN = _[2].to(self.gpu_id)
                uZ = _[3].to(self.gpu_id)
                outE, outN, outZ = self.model(a)
                loss_rel = loss_criterion((outE,outN,outZ), (uE,uN,uZ), self.loss_weights, relative=True)
                loss_abs = loss_criterion((outE,outN,outZ), (uE,uN,uZ), self.loss_weights, relative=False)
            
                train_losses_relative.update(loss_rel.item(), get_batch_size(a))
                train_losses_absolute.update(loss_abs.item(), get_batch_size(a))
            
                # compute gradients and update parameters
                self.optimizer.zero_grad()
                loss_rel.backward()
                self.optimizer.step()
        
            train_history['loss_relative'].append(train_losses_relative.avg)
            train_history['loss_absolute'].append(train_losses_absolute.avg)

            # validation
            self.model.eval()
            with torch.no_grad():
                val_losses_relative = RunningAverage()
                val_losses_absolute = RunningAverage()

                # training
                for _ in self.val_data:
                    a = _[0].to(self.gpu_id)
                    uE = _[1].to(self.gpu_id)
                    uN = _[2].to(self.gpu_id)
                    uZ = _[3].to(self.gpu_id)
                    outE, outN, outZ = self.model(a)
                    loss_rel_val = loss_criterion((outE,outN,outZ), (uE,uN,uZ), self.loss_weights, relative=True)
                    loss_abs_val = loss_criterion((outE,outN,outZ), (uE,uN,uZ), self.loss_weights, relative=False)

                    val_losses_relative.update(loss_rel_val.item(), get_batch_size(a))
                    val_losses_absolute.update(loss_abs_val.item(), get_batch_size(a))

                val_history['loss_relative'].append(val_losses_relative.avg)
                val_history['loss_absolute'].append(val_losses_absolute.avg)
                
                lr_scheduler.step(val_losses_relative.avg)

                t2 = timeit.default_timer()
                print(f'Epoch {ep+1}/{self.epochs}: {t2-t1:.2f}s - Training loss = {train_losses_relative.avg:.5f} - Validation loss = {val_losses_relative.avg:.5f}'\
                    f' - Training accuracy = {train_losses_absolute.avg:.5f} - Validation accuracy = {val_losses_absolute.avg:.5f}')

                # save the model
                if val_losses_relative.avg < best_loss and self.gpu_id == 0:
                    best_loss = val_losses_relative.avg
                    # ckp = model.state_dict()
                    ckp = self.model.module.state_dict() #DDP
                    torch.save(ckp, './logs/models/bestmodel-'+name_config+f'-epochs{self.epochs}.pt')
                
                if early_stopper.early_stop(val_losses_relative.avg):
                    break

                # save intermediate losses
                if ep%2==0 and self.rank == 0:
                    with h5py.File(f'./logs/loss/loss-{name_config}-epoch{ep}on{self.epochs}.h5', 'w') as f:
                        f.create_dataset('train_loss_relative', data=train_history['loss_relative'])
                        f.create_dataset('train_loss_absolute', data=train_history['loss_absolute'])
                        f.create_dataset('val_loss_relative', data=val_history['loss_relative'])
                        f.create_dataset('val_loss_absolute', data=val_history['loss_absolute'])

                    # remove the previous losses saved
                    if ep>2:
                        os.remove(f'./logs/loss/loss-{name_config}-epoch{ep-2}on{self.epochs}.h5')

                    last_epoch_saved = ep # to remove the last intermediate save at the end
    
        if self.rank == 0:
            # save the final loss
            with h5py.File(f'./logs/loss/loss-{name_config}-epochs{ep+1}.h5', 'w') as f:
                f.create_dataset('train_loss_relative', data=train_history['loss_relative'])
                f.create_dataset('train_loss_absolute', data=train_history['loss_absolute'])
                f.create_dataset('val_loss_relative', data=val_history['loss_relative'])
                f.create_dataset('val_loss_absolute', data=val_history['loss_absolute'])

            os.remove(f'./logs/loss/loss-{name_config}-epoch{last_epoch_saved}on{self.epochs}.h5')
            return last_epoch_saved
