use std::f32::consts::FRAC_PI_4;

use burn::{
    backend::NdArray,
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
    vision::Transform2D,
};
use rand::Rng;

#[derive(Clone, Debug, Default)]
pub struct MnistBatcher {}

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image))
            .map(|data| {
                Tensor::<NdArray, 2>::from_data(data.convert::<B::FloatElem>(), &Default::default())
            })
            .map(|tensor| tensor.reshape([1, 28, 28]))
            // normalize: make between [0,1] and make the mean =  0 and std = 1
            // values mean=0.1307,std=0.3081 were copied from Pytorch Mist Example
            // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .map(mangle_image_batch)
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<NdArray, 1, Int>::from_data(
                    TensorData::from([(item.label as i64).elem::<B::IntElem>()]),
                    &Default::default(),
                )
            })
            .collect();

        let images = Tensor::cat(images, 0);
        let images = Tensor::from_data(images.into_data(), device);

        let targets = Tensor::cat(targets, 0);
        let targets = Tensor::from_data(targets.into_data(), device);

        MnistBatch { images, targets }
    }
}

/// Mange the image by applying small random transformations to augment the dataset.
///
/// * `images` - The images with shape [batch size, height, width]
///
/// ## Return
///
/// The transformed images tensor with shape [batch size, height, width]
fn mangle_image_batch<B: Backend>(images: Tensor<B, 3>) -> Tensor<B, 3> {
    let mut rng = rand::rng();

    // Resample
    let shear = Transform2D::shear(
        rng.random_range(-0.6..0.6),
        rng.random_range(-0.6..0.6),
        0.0,
        0.0,
    );
    let rotation = Transform2D::rotation(rng.random_range(-FRAC_PI_4..FRAC_PI_4), 0.0, 0.0);
    let scale = Transform2D::scale(
        rng.random_range(0.6..1.5),
        rng.random_range(0.6..1.5),
        0.0,
        0.0,
    );
    let translate =
        Transform2D::translation(rng.random_range(-0.2..0.2), rng.random_range(-0.2..0.2));

    let transform = Transform2D::composed([translate, shear, scale, rotation]);

    transform
        .transform(images.unsqueeze_dim::<4>(1))
        .squeeze_dims::<3>(&[1])
}
