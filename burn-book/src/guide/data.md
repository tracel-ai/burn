# Data

Normaly you have to train your model on some sort of dataset.
Burn provides a library of very useful dataset sources and transformation.
There is an hugging face dataset utilities that allows you to download and store data from hugging face into a SQLite database for extrmelly efficient data streaming and storage.
For this guide, we will use the provided MNIST dataset in `burn_dataset::source::MNISTDataset`, interested users can look at how the implementation is done.

To iterate over a dataset efficiently, the `Dataloader` struct is also provided, we also need to implement the `Batcher` trait.
The goal of the batcher is to map individual dataset item into a batched tensor that can be used as input by our model defined previously.

```rust, ignore
use burn::{
    data::{dataloader::batcher::Batcher, dataset::source::huggingface::MNISTItem},
    tensor::{backend::Backend, Data, ElementConversion, Int, Tensor},
};

pub struct MNISTBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> MNISTBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

```

This codeblock define a batcher struct with the device in which the tensor should be send before being passed to the model.
Note that the device is an associative type of the `Backend` trait since not all backends exposes the sames devices.
As an example, the libtorch based backend exposes `Cuda(gpu_index)`, `Cpu`, `Vulkan` and `Metal` devices, but the ndarray only expose the `Cpu` device.

Next, we need to actually implement the batching logic.

```rust, ignore
#[derive(Clone, Debug)]
pub struct MNISTBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<MNISTItem, MNISTBatch<B>> for MNISTBatcher<B> {
    fn batch(&self, items: Vec<MNISTItem>) -> MNISTBatch<B> {
        let images = items
            .iter()
            .map(|item| Data::<f32, 2>::from(item.image))
            .map(|data| Tensor::<B, 2>::from_data(data.convert()))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            // normalize: make between [0,1] and make the mean =  0 and std = 1
            // values mean=0.1307,std=0.3081 were copied from Pytorch Mist Example
            // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data(Data::from([(item.label as i64).elem()])))
            .collect();

        let images = Tensor::cat(images, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        MNISTBatch { images, targets }
    }
}
```

In the previous example, we implement the `Batcher` trait with a list of`MNISTItem` as input and a single `MNISTBatch` as output.
The batch contains the images in the forme of a 3D Tensor and a targets tensors that contains the indexes of the correct digit class.
The first step is the parse the image array into a `Data` struct.
Burn provide the `Data` struct to encapsulate tensor storage information without being spefific for a backend.
When creating a tensor from data, we often need to convert the data precision to the current backend in use.
This can be done with the `.convert()` method while importing the `burn::tensor::elementConversion` trait.
This also allow you to call `.elem()` on a spefic number to convert it to the current backend element type in use.
