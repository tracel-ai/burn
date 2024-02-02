use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::vision::{ImageDatasetItem, ImageTarget, PixelDepth},
    },
    tensor::{backend::Backend, Data, Device, ElementConversion, Int, Shape, Tensor},
};

// CIFAR-10 mean and std values
const MEAN: [f32; 3] = [0.4914, 0.48216, 0.44653];
const STD: [f32; 3] = [0.24703, 0.24349, 0.26159];

/// Normalizer for the CIFAR-10 dataset.
pub struct Normalizer<B: Backend> {
    pub mean: Tensor<B, 4>,
    pub std: Tensor<B, 4>,
}

impl<B: Backend> Normalizer<B> {
    /// Creates a new normalizer.
    pub fn new(device: &Device<B>) -> Self {
        let mean = Tensor::from_floats(MEAN, device).reshape([1, 3, 1, 1]);
        let std = Tensor::from_floats(STD, device).reshape([1, 3, 1, 1]);
        Self { mean, std }
    }

    /// Normalizes the input image according to the CIFAR-10 dataset.
    ///
    /// The input image should be in the range [0, 1].
    /// The output image will be in the range [-1, 1].
    ///
    /// The normalization is done according to the following formula:
    /// `input = (input - mean) / std`
    pub fn normalize(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        (input - self.mean.clone()) / self.std.clone()
    }
}

pub struct ClassificationBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct ClassificationBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> ClassificationBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<ImageDatasetItem, ClassificationBatch<B>> for ClassificationBatcher<B> {
    fn batch(&self, items: Vec<ImageDatasetItem>) -> ClassificationBatch<B> {
        fn image_as_vec_u8(item: ImageDatasetItem) -> Vec<u8> {
            // Convert Vec<PixelDepth> to Vec<u8> (we know that CIFAR images are u8)
            item.image
                .into_iter()
                .map(|p: PixelDepth| -> u8 { p.try_into().unwrap() })
                .collect::<Vec<u8>>()
        }

        let targets = items
            .iter()
            .map(|item| {
                // Expect class label (int) as target
                if let ImageTarget::Label(y) = item.target {
                    Tensor::<B, 1, Int>::from_data(Data::from([(y as i64).elem()]), &self.device)
                } else {
                    panic!("Invalid target type")
                }
            })
            .collect();

        let images = items
            .into_iter()
            .map(|item| Data::new(image_as_vec_u8(item), Shape::new([32, 32, 3])))
            .map(|data| {
                Tensor::<B, 3>::from_data(data.convert(), &self.device)
                    // permute(2, 0, 1)
                    .swap_dims(2, 1) // [H, C, W]
                    .swap_dims(1, 0) // [C, H, W]
            })
            .map(|tensor| tensor / 255) // normalize between [0, 1]
            .collect();

        let images = Tensor::stack(images, 0);
        let targets = Tensor::cat(targets, 0);

        let normalizer = Normalizer::<B>::new(&self.device);
        let images = normalizer.normalize(images);

        ClassificationBatch { images, targets }
    }
}
