# Backend Extension

Burn aims to be the most flexible deep learning framework. While it's crucial to maintain
compatibility with a wide variety of backends, Burn provides the ability to extend the functionality
of a backend implementation to suit your modeling requirements. This versatility is advantageous in
numerous ways, such as supporting custom operations like flash attention or manually fusing
operations for enhanced performance.

In this section, we will go into the process of extending a backend, providing multiple examples.
But before we proceed, let's establish the fundamental principles that will empower you to craft
your own backend extensions.

As you can observe, most types in Burn are generic over the Backend trait. This might give the
impression that Burn operates at a high level over the backend layer. However, making the trait
explicit instead of being chosen via a compilation flag was a thoughtful design decision. This
explicitness does not imply that all backends must be identical; rather, it offers a great deal of
flexibility when composing backends. The autodifferentiation backend trait (see
[autodiff section](../../building-blocks/autodiff.md)) is an example of how the backend trait has
been extended to enable gradient computation with backpropagation. Furthermore, this design allows
you to create your own backend extension. To achieve this, you need to design your own backend trait
specifying which functions should be supported.

```rust, ignore
pub trait Backend: burn::tensor::backend::Backend {
    fn my_new_function(tensor: B::TensorPrimitive<2>) -> B::TensorPrimitive<2> {
        // You can define a basic implementation reusing the Burn Backend API.
        // This can be useful since all backends will now automatically support
        // your model. But performance can be improved for this new
        // operation by implementing this block in specific backends.
    }
}
```

You can then implement your new custom backend trait for any backend that you want to support:

```rust, ignore
impl<E: TchElement> Backend for burn_tch::LibTorch<E,F,I> {
   fn my_new_function(tensor: TchTensor<E, 2>) -> TchTensor<E, 2> {
      // My Tch implementation
   }
}

impl<E: NdArrayElement> Backend for burn_ndarray::NdArray<E> {
    // No specific implementation, but the backend can still be used.
}
```

You can support the backward pass using the same pattern.

```rust, ignore
impl<B: Backend> Backend for burn_autodiff::Autodiff<B> {
    // No specific implementation; autodiff will work with the default
    // implementation. Useful if you still want to train your model, but
    // observe performance gains mostly during inference.
}

impl<B: Backend> Backend for burn_autodiff::Autodiff<B> {
   fn my_new_function(tensor: AutodiffTensor<E, 2>) -> AutodiffTensor<E, 2> {
      // My own backward implementation, generic over my custom Backend trait.
      //
      // You can add a new method `my_new_function_backward` to your custom backend
      // trait if you want to invoke a custom kernel during the backward pass.
   }
}

impl<E: TchElement> Backend for burn_autodiff::Autodiff<burn_tch::LibTorch<E,F,I>> {
   fn my_new_function(tensor: AutodiffTensor<E, 2>) -> AutodiffTensor<E, 2> {
      // My own backward implementation, generic over a backend implementation.
      //
      // This is another way to call a custom kernel for the backward pass that
      // doesn't require the addition of a new `backward` function in the custom backend.
      // This is useful if you don't want all backends to support training, reducing
      // the need for extra code when you know your model will only be trained on one
      // specific backend.
   }
}
```

The specifics of each implementation will be covered by the examples provided in this section. The
`cubecl` compiler frontend is the recommended method of implementing custom kernels, since it
supports multiple backends, including `wgpu` and `CUDA`, and is the way first-party `burn` kernels
are written.
