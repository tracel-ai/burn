# Einsum demo

Run from the workspace root:

```sh
cargo run -p burn-tensor --example einsum --features flex,autodiff
```

The demo exercises literal equations, a runtime equation with mixed operand
ranks, trace, MaskFormer-style mask prediction, and automatic differentiation.
Expected output:

```text
Matrix multiplication: [19.0, 22.0, 43.0, 50.0]
Runtime equation: [50.0, 110.0]
Trace: [5.0]
Mask prediction [1, 2, 1, 2]: [7.0, 10.0, 15.0, 22.0]
Gradient of dot(x, x): [2.0, 4.0, 6.0]
```

`einsum!` parses literal equations at compile time and generates checked,
left-to-right contraction stages. `Tensor::einsum` accepts runtime strings.
Both use the same parser and executor and support broadcasting, ellipses,
diagonals, and multiple operands. Scalar tensors use shape `[1]`.

The executor follows the [Python reference](https://github.com/Mikyx-1/pytorch-einsum-reference).
It uses existing Burn operations and does not search for an optimized contraction
order. Float and Int operands must share their dtype and device; quantized
operands are unsupported.
