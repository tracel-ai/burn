
# Backend

The Backend trait abstracts multiple things:

- Device type
- Float tensor type
- Bool tensor type
- Int tensor type
- Float element type
- Int element type
- Float tensor operations (kernels)
- Int tensor operations (kernels)
- Bool tensor operations (kernels)

Even though having one type for tensors is convenient for the tensor API, it can be cumbersome when implementing a backend.
Therefore, backends can decide, through associated types, what types they want to use for their int, float, and bool tensors.
Since float and int can have multiple precisions, the float and int element types are also associated types that must be declared by the backend.

Note that the backend chooses the precision and not the user.
Since not all backends will support the same element types, no assumptions must be made.
Therefore, there are no methods on tensors to change the precision, except for the `to_full_precision` function, which ensures numerical stability on the current backend.
Backend implementations can provide a way to choose the precision, which can be accomplished with a generic parameter (e.g. `NdArray<f32>`).

To be as general as possible, tensor operations are implemented as plain functions.
There is no object or self, just functions that take tensors as input and often return tensors as output as well.
Backend implementations are free to use their own patterns to implement these kernels.
Note that Burn is a dynamic graph deep learning framework, so backends may have to implement asynchronous kernel executions for performance reasons.

## Autodiff

As of now, there is only one backend decorator that supports autodiff.
It follows the decorator pattern, making any backend differentiable.
However, the `AutodiffBackend` trait abstracts how gradients are calculated, and other approaches to autodiff might be added later.
For more information about how the current autodiff backend works, you can read this [blog post](https://burn.dev/blog/burn-rusty-approach-to-tensor-handling).
