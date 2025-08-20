# burn-collective

Collective operations on tensors

The following collective operation are implemented:

- `all-reduce`
    Aggregates a tensor between all peers, and distributes the result to all peers.
    Different strategies can be used on the local and global levels. The result can only be
    returned when all peers have called the all-reduce.
- `reduce`
    Aggregates a tensor from all peers onto one peer, called the "root"
- `broadcast`
    Copies a tensor from one peer to all other peers in the collective.

Peers must call `register` before calling any other operation.
The total number of devices on the node, or nodes in the collective, must be known ahead of time.

In many libraries like NCCL and PyTorch, participating units are called "ranks".
This name is confusing in the context of tensors, so in burn-collective the participating units
are called "peers".

*`reduce` and `broadcast` are not yet implemented for multi-node contexts*

## Local and Global

Internally, there are two levels to the collective operations: local and global. Operations are done on the local level, then optionally on the global level.

| Local                                      | Global                                        |
|-----------------------------------------------|-----------------------------------------------|
| Intra-node (typically within one machine)     | Inter-node (typically across machies)         |
| Participants are threads (one per peer/GPU) | Participants are processes (one per node)     |
| Communication depends on backend              | Network peer-to-peer communication            |
| Local server is launched automatically      | Global coordinator must be launched manually  |
| Local server does the aggregation          | Nodes do the operations themselves            |

For global operations (ie. with multiple nodes), there must be a global orchestrator available.
Start one easily with `burn_collective::start_global_orchestrator()`.

On the global level, nodes use the `burn_communication::data_service::TensorDataService` to
expose and download tensors in a peer-to-peer manner, in order to be independent.

## Components

The following are the important pieces of the collective operations system.

| Term                           | One per...    | Meaning
|--------------------------------|---------------|----------------------------------------------------------
| Local Collective Client        | Peer/thread | Requests operations to the Local Collective Server
| Local Collective Server        | Node/process  | Does local-level ops for threads in this process. In the case of global operations, passes operations on to the Global Collective Client.
| Global Collective Client       | Node/process  | Does global-level ops for this node. Registers and requests strategies from the Global Collective Orchestrator.
| Global Collective Orchestrator | Collective    | Responds to the Global Collective Client from each node. Responsible for aggregation strategies.

## Strategies

Different strategies can be used on the local and global level.

### Centralized

An arbitrary peer is designated as the "root", and all others are transferred to the root's device.
The operation is done on that device.
The resulting tensor then sent to each peer.

### Tree

Tensors in groups of N are aggregated together. This is done recursively until only one tensor
remains. The strategy tries to put devices of the same type closer in the tree.
When N=2, this is like a binary tree reduce.
The resulting tensor then sent to each peer

### Ring

See this good explanation: <https://blog.dailydoseofds.com/p/all-reduce-and-ring-reduce-for-model>

The tensors are sliced into N parts, where N is the number of tensors to aggregate.
Then, the slices are sent around in a series of cycles and aggregated until every tensor's slices
is a sum of the other corresponding slices.

In the case where the tensors are too small to split into N slices, a fallback algorithm is used.
For now, the fallback is a binary tree.

(p=3, n=3)

o->o  o  
o  o->o  
o  o  o->

o  1->o  
o  o  1->
1->o  o  

o  1  2->
2->o  1  
1  2->o  

3  1  2
2  3  1
1  2  3

(This is essentially a reduce-scatter)

3->x  x  
x  3->x  
x  x  3->

3  3->x  
x  3  3->
3->x  3  

3  3  3->
3->3  3  
3  3->3  

3  3  3
3  3  3
3  3  3

(This is essentially an all-gather)

This is done so that every peer is both sending and receiving data at any moment.
This is an important part of this strategy's advantages.

The ring strategy takes full advantage of the bandwidth available. The latency scales with the
number of peers.

So when the tensors are very small, or when the number of peers is very large, the latency is more
important in the ring strategy, and a tree algorithm is better. Otherwise, the ring algorithm is
the better.

In multi-node contexts, use of the Ring strategy in the local level may be less
advantageous. With multiple nodes, the global all-reduce step is enabled, and its result
is redistributed to all devices.
The Ring strategy inherently distributes the result, which in this context would not be necessary.

It is recommended to use the Ring strategy at the global level

### Double binary tree

<https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/>
