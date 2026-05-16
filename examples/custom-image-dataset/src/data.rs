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
pub struct Normalizer {
    pub mean: Tensor<4>,
    pub std: Tensor<4>,
}

impl Normalizer {
    /// Creates a new normalizer.
    pub fn new(device: &Device) -> Self {
        let mean = Tensor::<1>::from_floats(MEAN, device).reshape([1, 3, 1, 1]);
        let std = Tensor::<1>::from_floats(STD, device).reshape([1, 3, 1, 1]);
        Self { mean, std }
    }

    /// Normalizes the input image according to the CIFAR-10 dataset.
    ///
    /// The input image should be in the range [0, 1].
    /// The output image will be in the range [-1, 1].
    ///
    /// The normalization is done according to the following formula:
    /// `input = (input - mean) / std`
    pub fn normalize(&self, input: Tensor<4>) -> Tensor<4> {
        (input - self.mean.clone()) / self.std.clone()
    }

    /// Returns a new normalizer on the given device.
    pub fn to_device(&self, device: &Device) -> Self {
        Self {
            mean: self.mean.clone().to_device(device),
            std: self.std.clone().to_device(device),
        }
    }
}

#[derive(Clone)]
pub struct ClassificationBatcher {
    normalizer: Normalizer,
}

#[derive(Clone, Debug)]
pub struct ClassificationBatch {
    pub images: Tensor<4>,
    pub targets: Tensor<1, Int>,
    pub images_path: Vec<String>,
}

impl ClassificationBatcher {
    pub fn new(device: &Device) -> Self {
        Self {
            normalizer: Normalizer::new(device),
        }
    }
}

impl Batcher<ImageDatasetItem, ClassificationBatch> for ClassificationBatcher {
    fn batch(&self, items: Vec<ImageDatasetItem>, device: &Device) -> ClassificationBatch {
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
                    Tensor::<1, Int>::from_data(TensorData::from([y as i32]), device)
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
                Tensor::<3>::from_data(data, device)
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
