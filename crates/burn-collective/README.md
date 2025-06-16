# burn-collective

Tools for collective operations on tensors

The collective operations are the following:
- `all-reduce`


Threads can register to use collective operations with `register()`

When a collective operation is called N times (N calls to `register`), 
the aggregator starts the collective operation. Every tensor passed is aggregated.  

## Stategies

### Centralized

An arbitrary tensor is designated as the base, and all others are transfered to the base's device.
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

The ring strategy takes full advantage of the bandwith available. The latency scales with the 
number of devices. 

So when the tensors are very small, or when the number of devices is very large, the latency is more 
important in the ring strategy, and a tree algorithm is better. Otherwise, the ring algorithm is 
the better.


### Double binary tree

https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/

