# burn-collective

Collective operations on tensors

There is only one collective operation so far:
- `all-reduce`
    Aggregates a tensor between all deveices, and distributes the result to all devices. 
    Different strategies can be used on the local and global levels. The result can only be 
    returned when all devices have called the all-reduce.

Threads must call `register` before calling any other operation. 
The total number of devices on the node, or nodes in the collective, must be known ahead of time.

## Local and Global

There are two levels to the collective operations: local and global. Operations are done on the local level, then optionally on the global level.

| Local                                  	| Global                                        |
|----------------------------------------	|-----------------------------------------------|
| Peers are threads (one per device)     	| Peers are processes (one per node)            |
| Communication depends on backend          | Network peer-to-peer communication            |
| Local server is launched automatically 	| Global coordinator must be launched manually  |
| Local server does the aggregation     	| Nodes do the operations themselves            |

For global operations, there must be a global orchestrator available. 
Start one easily with `burn_collective::start_global_orchestrator()`.

## Components

The following are the important pieces of the collective operations system.

| Term                           | One per...    | Meaning                                                  
|--------------------------------|---------------|----------------------------------------------------------
| Local Collective Client        | Device/thread | Requests operations to the Local Collective Server
| Local Collective Server        | Node/process  | Does local-level ops for threads in this process. In the case of global operations, passes operations on to the Global Collective Client.
| Global Collective Client       | Node/process  | Does global-level ops for this node. Registers and requests strategies from the Global Collective Orchestrator.
| Global Collective Orchestrator | Collective    | Responds to the Global Collective Client from each node. Responsible for aggregation strategies.

## Strategies

Different strategies can be used on the local and global level.

### Centralized

An arbitrary tensor is designated as the base, and all others are transferred to the base's device.
The operation is done on that device.
The resulting tensor then sent to the device corresponding to each original tensor.

### Tree

Tensors in groups of N are aggregated together. This is done recursively until only one tensor 
remains. For now, the grouping strategy is unaware of the devices.
When N=2, this is like a binary tree reduce.
The resulting tensor then sent to the device corresponding to each original tensor.

### Ring

See this good explanation: https://blog.dailydoseofds.com/p/all-reduce-and-ring-reduce-for-model

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

This is done so that every device is both sending and receiving data at any moment. 
This is an important part of this strategy's advantages.

The ring strategy takes full advantage of the bandwidth available. The latency scales with the 
number of devices. 

So when the tensors are very small, or when the number of devices is very large, the latency is more 
important in the ring strategy, and a tree algorithm is better. Otherwise, the ring algorithm is 
the better.


### Double binary tree

https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/

