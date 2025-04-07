use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::vision::{Annotation, ImageDatasetItem, PixelDepth},
    },
    prelude::*,
};

// CIFAR-10 mean and std values
const MEAN: [f32; 3] = [0.4914, 0.48216, 0.44653];
const STD: [f32; 3] = [0.24703, 0.24349, 0.26159];

/// Normalizer for the CIFAR-10 dataset.
#[derive(Clone)]
pub struct Normalizer<B: Backend> {
    pub mean: Tensor<B, 4>,
    pub std: Tensor<B, 4>,
}

impl<B: Backend> Normalizer<B> {
    /// Creates a new normalizer.
    pub fn new(device: &Device<B>) -> Self {
        let mean = Tensor::<B, 1>::from_floats(MEAN, device).reshape([1, 3, 1, 1]);
        let std = Tensor::<B, 1>::from_floats(STD, device).reshape([1, 3, 1, 1]);
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

    /// Returns a new normalizer on the given device.
    pub fn to_device(&self, device: &B::Device) -> Self {
        Self {
            mean: self.mean.clone().to_device(device),
            std: self.std.clone().to_device(device),
        }
    }
}

#[derive(Clone)]
pub struct ClassificationBatcher<B: Backend> {
    normalizer: Normalizer<B>,
}

#[derive(Clone, Debug)]
pub struct ClassificationBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
    pub images_path: Vec<String>,
}

impl<B: Backend> ClassificationBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            normalizer: Normalizer::<B>::new(&device),
        }
    }
}

impl<B: Backend> Batcher<B, ImageDatasetItem, ClassificationBatch<B>> for ClassificationBatcher<B> {
    fn batch(&self, items: Vec<ImageDatasetItem>, device: &B::Device) -> ClassificationBatch<B> {
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
                if let Annotation::Label(y) = item.annotation {
                    Tensor::<B, 1, Int>::from_data(
                        TensorData::from([(y as i64).elem::<B::IntElem>()]),
                        device,
                    )
                } else {
                    panic!("Invalid target type")
                }
            })
            .collect();

        // Original sample path
        let images_path: Vec<String> = items.iter().map(|item| item.image_path.clone()).collect();

        let images = items
            .into_iter()
            .map(|item| TensorData::new(image_as_vec_u8(item), Shape::new([32, 32, 3])))
            .map(|data| {
                Tensor::<B, 3>::from_data(data.convert::<B::FloatElem>(), device)
                    // permute(2, 0, 1)
                    .swap_dims(2, 1) // [H, C, W]
                    .swap_dims(1, 0) // [C, H, W]
            })
            .map(|tensor| tensor / 255) // normalize between [0, 1]
            .collect();

        let images = Tensor::stack(images, 0);
        let targets = Tensor::cat(targets, 0);

        let images = self.normalizer.to_device(device).normalize(images);

        ClassificationBatch {
            images,
            targets,
            images_path,
        }
    }
}
