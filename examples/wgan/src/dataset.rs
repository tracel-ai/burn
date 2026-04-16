use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
};

#[derive(Clone, Debug, Default)]
pub struct MnistBatcher {}

#[derive(Clone, Debug)]
pub struct MnistBatch {
    pub images: Tensor<4>,
    pub targets: Tensor<1, Int>,
}

impl Batcher<MnistItem, MnistBatch> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &Device) -> MnistBatch {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image))
            .map(|data| Tensor::<2>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            // Set std=0.5 and mean=0.5 to keep consistent with pytorch WGAN example
            .map(|tensor| ((tensor / 255) - 0.5) / 0.5)
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<1, Int>::from_data(TensorData::from([item.label as i64]), device))
            .collect();

        let images = Tensor::stack(images, 0);
        let targets = Tensor::cat(targets, 0);

        MnistBatch { images, targets }
    }
}
