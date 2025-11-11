use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
    tensor::Int,
};

#[derive(Clone, Debug, Default)]
pub struct MnistBatcher {}

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        let images: Vec<Tensor<B, 3>> = items
            .iter()
            .map(|item| {
                let data = TensorData::from(item.image);
                let tensor = Tensor::<B, 2>::from_data(data.convert::<f32>(), device);
                // Normalize: make between [0,1] and make the mean = 0 and std = 1
                // Values mean=0.1307, std=0.3081 were copied from PyTorch MNIST example
                let tensor = ((tensor / 255.0) - 0.1307) / 0.3081;
                tensor.reshape([1, 28, 28])
            })
            .collect();

        let targets: Vec<Tensor<B, 1, Int>> = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    TensorData::from([(item.label as i64).elem::<B::IntElem>()]),
                    device,
                )
            })
            .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        MnistBatch { images, targets }
    }
}
