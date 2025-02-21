# MNIST no-std

This example demonstrates how to train and perform inference in a `no-std`
environment.

## Running

There are two examples in this crate:

1. Training

    Trains a new model and exports it to the given path.

    ``` shell
    cargo run --release --example train
    ```

    This example downloads the MNIST dataset, trains a new model, and outputs
    the model to the given path(default: `model.bin`).

    You can run `cargo run --release --example train -- --help` for detailed
    usage.

2. Inference

    Loads a model from the given path, tests it with a given image, and prints
    the inference result.

    ```shell
    # cargo run --release --example infer -- --binary-path=samples/8.bin
    cargo run --release --example infer -- -i ${binary_path}
    ```

    This command loads the model the model from the given
    path(default: `model.bin`) and tests it with the given binary, and prints
    the inference result. For convenience, you can use the sample binaries in
    the `samples` folder.

    You can run `cargo run --release --example infer -- --help` for detailed
    usage.

## Design

The crate is `no-std` and contains only logic related to training and inference.
It provides APIs that accept only primitive types as parameters to ensure
portability. It is the caller's responsibility to provide the data and control
the workflow.

The crate consist of 3 modules:

1. proto

    A module that contains the proto definitions shared between the crate and
    its caller. It only includes primitive types to demonstrate portability.

2. train

    A module that contains a simple `Trainer` and a public module named
    `no_std_world`, which simulates a `no-std` environment and can be called
    externally.

    It exports the following APIs:

    * initialize: Initializes a global trainer with a given random seed and
                  learning rate.
    * train: Trains the model with the given data and return the loss and
             accuracy for feedback.
    * valid: Validates the model with the given data and return the loss and
             accuracy for feedback.
    * export: Exports the model as bytes so it can be persisted.

    You can refer to `examples/train.rs` for usage.

3. inference

    A module that contains a simple `Model` and a public module named
    `no_std_world`, which simulates a `no-std` environment, and can be called
    externally.

    It exports the following APIs:

    * initialize: Initializes a global model with the provided record bytes.
    * infer: Use the global model to perform inference with the given image and
             return its inference result.

    You can refer to `examples/infer.rs` for usage.
