## DDP
Distributed Data Parallel

The DDP is a learning strategy that trains a replica of the model on each device.

The DDP launches threads for each local device. Each thread on each node will run the model.
After the forward and backward passes, the gradients are synced between all peers on all nodes 
with an `all-reduce` operation.

While the DDP launches threads for each local device, it is the user's responsibility to launch the 
DDP on each node, and assure the collective configuration matches.

## Main device vs secondary devices 

The main device is responsible for validation, as well as event processing, which is used in the UI.

The first device is chosen as the main device.
