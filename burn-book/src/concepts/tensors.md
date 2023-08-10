# Tensors

At the core of burn lies the `Tensor` struct, which encompasses multiple types of tensors, including
`Float`, `Int`, and `Bool`. The element types of these tensors are specified by the backend and are
usually designated as a generic argument (e.g., `NdArrayBackend<f32>`). Although the same struct is
used for all tensors, the available methods differ depending on the tensor kind. You can specify the
desired tensor kind by setting the third generic argument, which defaults to `Float`. The first
generic argument specifies the backend, while the second specifies the number of dimensions.

```rust
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, Int};

fn function<B: Backend>(tensor_float: Tensor<B, 2>) {
    let _tensor_bool = tensor_float.clone().equal_elem(2.0); // Tensor<B, 2, Bool>
    let _tensor_int = tensor_float.argmax(1); // Tensor<B, 2, Int>
}
```

As demonstrated in the previous example, nearly all operations require owned tensors as parameters,
which means that calling `Clone` explicitly is necessary when reusing the same tensor multiple
times. However, there's no need to worry since the tensor's data won't be copied, it will be flagged
as readonly when multiple tensors use the same allocated memory. This enables backends to reuse
tensor data when possible, similar to a copy-on-write pattern, while remaining completely
transparent to the user.
