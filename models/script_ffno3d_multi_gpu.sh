#!/bin/bash

master_hostname=$1

python3 ddp_main.py @batch_size=32 @nlayers=8 \
@list_D1 32 32 32 32 32 32 32 32 \
@list_D2 32 32 32 32 32 32 32 32 \
@list_D3 64 64 128 128 256 256 320 320 \
@list_M1 16 16 16 16 16 16 16 16 \
@list_M2 16 16 16 16 16 16 16 16 \
@list_M3 16 32 32 32 32 32 32 32 \
@learning_rate 0.006 \
@master_hostname=$master_hostname \
@model_type=ffno3d
