# Data

Typically, one trains a model on some dataset. Burn provides a library of very useful dataset
sources and transformations, such as Hugging Face dataset utilities that allow to download and store
data into an SQLite database for extremely efficient data streaming and storage. For this guide
though, we will use the MNIST dataset from `burn::data::dataset::vision` which requires no external
dependency.

To iterate over a dataset efficiently, we will define a struct which will implement the `Batcher`
trait. The goal of a batcher is to map individual dataset items into a batched tensor that can be
used as input to our previously defined model.

Let us start by defining our dataset functionalities in a file `src/data.rs`. We shall omit some of
the imports for brevity, but the full code for following this guide can be found at
`examples/guide/` [directory](https://github.com/tracel-ai/burn/tree/main/examples/guide).

```rust , ignore
use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
};


#[derive(Clone, Default)]
pub struct MnistBatcher {}
```

This batcher is pretty straightforward, as it only defines a struct that will implement the
`Batcher` trait. The trait is generic over the `Backend` trait, which includes an associated type
for the device, as not all backends expose the same devices. As an example, the Libtorch-based
backend exposes `Cuda(gpu_index)`, `Cpu`, `Vulkan` and `Metal` devices, while the ndarray backend
only exposes the `Cpu` device.

Next, we need to actually implement the batching logic.

```rust , ignore
# use burn::{
#     data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
#     prelude::*,
# };
#
# #[derive(Clone, Default)]
# pub struct MnistBatcher {}
#
#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 2>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            // Normalize: scale between [0,1] and make the mean=0 and std=1
            // values mean=0.1307,std=0.3081 are from the PyTorch MNIST example
            // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data([(item.label as i64).elem::<B::IntElem>()], device)
            })
            .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        MnistBatch { images, targets }
    }
}
```

<details>
<summary><strong>🦀 Iterators and Closures</strong></summary>

The iterator pattern allows you to perform some tasks on a sequence of items in turn.

In this example, an iterator is created over the `MnistItem`s in the vector `items` by calling the
`iter` method.

_Iterator adaptors_ are methods defined on the `Iterator` trait that produce different iterators by
changing some aspect of the original iterator. Here, the `map` method is called in a chain to
transform the original data before consuming the final iterator with `collect` to obtain the
`images` and `targets` vectors. Both vectors are then concatenated into a single tensor for the
current batch.

You probably noticed that each call to `map` is different, as it defines a function to execute on
the iterator items at each step. These anonymous functions are called
[_closures_](https://doc.rust-lang.org/book/ch13-01-closures.html) in Rust. They're easy to
recognize due to their syntax which uses vertical bars `||`. The vertical bars capture the input
variables (if applicable) while the rest of the expression defines the function to execute.

If we go back to the example, we can break down and comment the expression used to process the
images.

```rust, ignore
let images = items                                                       // take items Vec<MnistItem>
    .iter()                                                              // create an iterator over it
    .map(|item| TensorData::from(item.image).convert::<B::FloatElem>())  // for each item, convert the image to float data struct
    .map(|data| Tensor::<B, 2>::from_data(data, device))                 // for each data struct, create a tensor on the device
    .map(|tensor| tensor.reshape([1, 28, 28]))                           // for each tensor, reshape to the image dimensions [C, H, W]
    .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)                    // for each image tensor, apply normalization
    .collect();                                                          // consume the resulting iterator & collect the values into a new vector
```

For more information on iterators and closures, be sure to check out the
[corresponding chapter](https://doc.rust-lang.org/book/ch13-00-functional-features.html) in the Rust
Book.

</details><br>

In the previous example, we implement the `Batcher` trait with a list of `MnistItem` as input and a
single `MnistBatch` as output. The batch contains the images in the form of a 3D tensor, along with
a targets tensor that contains the indexes of the correct digit class. The first step is to parse
the image array into a `TensorData` struct. Burn provides the `TensorData` struct to encapsulate
tensor storage information without being specific for a backend. When creating a tensor from data,
we often need to convert the data precision to the current backend in use. This can be done with the
`.convert()` method (in this example, the data is converted backend's float element type
`B::FloatElem`). While importing the `burn::tensor::ElementConversion` trait, you can call `.elem()`
on a specific number to convert it to the current backend element type in use.
