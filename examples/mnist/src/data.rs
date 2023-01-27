use burn::{
    data::{dataloader::batcher::Batcher, dataset::source::huggingface::MNISTItem},
    tensor::{backend::Backend, Data, Tensor},
};

pub struct MNISTBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct MNISTBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B::IntegerBackend, 1>,
}

impl<B: Backend> MNISTBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<MNISTItem, MNISTBatch<B>> for MNISTBatcher<B> {
    fn batch(&self, items: Vec<MNISTItem>) -> MNISTBatch<B> {
        let images = items
            .iter()
            .map(|item| Data::<f32, 2>::from(item.image))
            .map(|data| Tensor::<B, 2>::from_data(data.convert()))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            .map(|tensor| tensor / 255)
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B::IntegerBackend, 1>::from_data(Data::from([item.label as i64])))
            .collect();

        let images = Tensor::cat(images, 0).to_device(&self.device).detach();
        let targets = Tensor::cat(targets, 0).to_device(&self.device).detach();

        MNISTBatch { images, targets }
    }
}
