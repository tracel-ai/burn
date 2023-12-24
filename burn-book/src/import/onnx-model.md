# Import ONNX Model

## Why Importing Models is Necessary

In the realm of deep learning, it's common to switch between different frameworks depending on your
project's specific needs. Maybe you've painstakingly fine-tuned a model in TensorFlow or PyTorch and
now you want to reap the benefits of Burn's unique features for deployment or further testing. This
is precisely the scenario where importing models into Burn can be a game-changer.

## Traditional Methods: The Drawbacks

If you've been working with other deep learning frameworks like PyTorch, it's likely that you've
exported model weights before. PyTorch, for instance, lets you save model weights using its
`torch.save()` function. Yet, to port this model to another framework, you face the arduous task of
manually recreating the architecture in the destination framework before loading in the weights. Not
only is this method tedious, but it's also error-prone and hinders smooth interoperability between
frameworks.

It's worth noting that for models using cutting-edge, framework-specific features, manual porting
might be the only option, as standards like ONNX might not yet support these new innovations.

## Enter ONNX

[ONNX (Open Neural Network Exchange)](https://onnx.ai/onnx/intro/index.html) is designed to solve
such complications. It's an open-standard format that exports both the architecture and the weights
of a deep learning model. This feature makes it exponentially easier to move models between
different frameworks, thereby significantly aiding interoperability. ONNX is supported by a number
of frameworks including but not limited to TensorFlow, PyTorch, Caffe2, and Microsoft Cognitive
Toolkit.

### Advantages of ONNX

ONNX stands out for encapsulating two key elements:

1. **Model Information**: It captures the architecture, detailing the layers, their connections, and
   configurations.
2. **Weights**: ONNX also contains the trained model's weights.

This dual encapsulation not only simplifies the porting of models between frameworks but also allows
seamless deployment across different environments without compatibility concerns.

## Burn's ONNX Support: Importing Made Easy

Understanding the important role that ONNX plays in the contemporary deep learning landscape, Burn
simplifies the process of importing ONNX models via an intuitive API designed to mesh well with
Burn's ecosystem.

Burn's solution is to translate ONNX files into Rust source code as well as Burn-compatible weights.
This transformation is carried out through the burn-import crate's code generator during build time,
providing advantages for both executing and further training ONNX models.

### Advantages of Burn's ONNX Approach

1. **Native Integration**: The generated Rust code is fully integrated into Burn's architecture,
   enabling your model to run on various backends without the need for a separate ONNX runtime.

2. **Trainability**: The imported model is not just for inference; it can be further trained or
   fine-tuned using Burn's native training loop.

3. **Portability**: As the model is converted to Rust source code, it can be compiled into
   WebAssembly for browser execution. Likewise, this approach is beneficial for no-std embedded
   devices.

4. **Optimization**: Rust's compiler can further optimize the generated code for target
   architectures, thereby improving performance.

### Sample Code for Importing ONNX Model

Below is a step-by-step guide to importing an ONNX model into a Burn-based project:

#### Step 1: Update `build.rs`

Include the `burn-import` crate and use the following Rust code in your `build.rs`:

```rust, ignore
use burn_import::onnx::ModelGen;

fn main() {
    // Generate Rust code from the ONNX model file
    ModelGen::new()
        .input("src/model/mnist.onnx")
        .out_dir("model/")
        .run_from_script();
}
```

#### Step 2: Modify `mod.rs`

Add this code to the `mod.rs` file located in `src/model`:

```rust, ignore
pub mod mnist {
    include!(concat!(env!("OUT_DIR"), "/model/mnist.rs"));
}
```

#### Step 3: Utilize Imported Model

Here's how to use the imported model in your application:

```rust, ignore
mod model;

use burn::tensor;
use burn_ndarray::{NdArray, NdArrayDevice};
use model::mnist::Model;

fn main() {
    // Initialize a new model instance
    let device = NdArrayDevice::default();
    let model: Model<NdArray<f32>> = Model::new(&device);

    // Create a sample input tensor (zeros for demonstration)
    let input = tensor::Tensor::<NdArray<f32>, 4>::zeros([1, 1, 28, 28], &device);

    // Perform inference
    let output = model.forward(input);

    // Print the output
    println!("{:?}", output);
}
```

### Working Examples

For practical examples, please refer to:

1. [MNIST Inference Example](https://github.com/tracel-ai/burn/tree/main/examples/onnx-inference)
2. [SqueezeNet Image Classification](https://github.com/tracel-ai/models/tree/main/squeezenet-burn)

By combining ONNX's robustness with Burn's unique features, you'll have the flexibility and power to
streamline your deep learning workflows like never before.

---

> 🚨**Note**: `burn-import` crate is in active development and currently supports a
> [limited set of ONNX operators](https://github.com/tracel-ai/burn/blob/main/burn-import/SUPPORTED-ONNX-OPS.md).
