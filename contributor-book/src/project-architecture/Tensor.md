# Tensor

A proper deep learning framework should have a fast tensor implementation with autodiff support, and Burn is no exception.
The tensor API abstracts away backend implementation details and focuses on usability without compromising performance.
To make it as easy as possible to use, there is only one tensor type, which is different from multiple tensor and deep learning crates in Rust.
Generic parameters are used instead to specialize the tensor type.

- **B: Backend:**
  The first argument is the backend on which the tensor implementation lies.
- **const D: usize:**
  The second argument is the dimensionality of the tensor.
- **K: TensorKind:**
  The third argument is the tensor kind, which can be either Float, Int or Bool.
  By default, the tensor kind is set to Float, so for most tensors, the kind argument is not necessary.

Having one struct for tensors reduces the complexity of the tensor API, which also means less duplicated documentation to write and maintain.

Tensors are thread-safe, which means that you can send a tensor to another thread, and everything will work, including auto-differentiation.
Note that there are no in-place tensor operations since all tensor operations take owned tensors as parameters, which make it possible to mutate them.
Tensors can be shared simply by cloning them, but if there is only one reference to a tensor, the backend implementation is free to reuse the tensor's allocated data.
For more information about how it is done, you can have a look at this [blog post](https://burn.dev/blog/burn-rusty-approach-to-tensor-handling).