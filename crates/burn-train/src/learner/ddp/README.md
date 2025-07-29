## DDP
Distributed Data Parallel

The DDP is a wrapper over the Learner that implements multi-threaded, multi-node learning with
the burn-collective library.

The DDP launches threads for each local device. Each thread on each node will run the model.
After the forward and backward passes, the gradients are synced between all peers on all nodes 
with an `all-reduce` operation.

While the DDP launches threads for each local device, it is the user's responsibility to launch the 
DDP on each node, and assure the collective configuration matches.

## Main device vs helper devices 

Each node has one main device, and helper devices. 

The difference between the main device and the helper devices is that the main device is the one
used to display metrics, process events, and do validation.

The helper devices only do training (no validation).
