use std::{f32::consts::FRAC_PI_4, fmt::Display};

use burn::{
    backend::NdArray,
    data::{
        dataloader::batcher::Batcher,
        dataset::{transform::Mapper, vision::MnistItem},
    },
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

impl<B: Backend> Batcher<B, MnistItemPrepared, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItemPrepared>, device: &B::Device) -> MnistBatch<B> {
        let images = items.iter().map(|item| item.image.clone()).collect();

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

#[derive(Clone, Debug, Copy)]
pub enum Transform {
    Translate,
    Shear,
    Scale,
    Rotation,
}

impl Display for Transform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Transform::Translate => f.write_str("Tr"),
            Transform::Shear => f.write_str("Sr"),
            Transform::Scale => f.write_str("Sc"),
            Transform::Rotation => f.write_str("Rot"),
        }
    }
}

#[derive(Default)]
pub struct MnistMapper {
    transforms: Vec<Transform>,
}

impl MnistMapper {
    pub fn transform(mut self, transforms: &[Transform]) -> Self {
        for t in transforms {
            self.transforms.push(*t);
        }
        self
    }
    pub fn translate(mut self) -> Self {
        self.transforms.push(Transform::Translate);
        self
    }
    pub fn shear(mut self) -> Self {
        self.transforms.push(Transform::Shear);
        self
    }
    pub fn scale(mut self) -> Self {
        self.transforms.push(Transform::Scale);
        self
    }
    pub fn rotation(mut self) -> Self {
        self.transforms.push(Transform::Rotation);
        self
    }
}

impl Mapper<MnistItem, MnistItemPrepared> for MnistMapper {
    fn map(&self, item: &MnistItem) -> MnistItemPrepared {
        prepare_image(&self.transforms, item.clone())
    }
}

#[derive(Clone, Debug)]
pub struct MnistItemPrepared {
    image: Tensor<NdArray, 3>,
    label: u8,
}

fn prepare_image(transforms: &[Transform], item: MnistItem) -> MnistItemPrepared {
    let data = TensorData::from(item.image);
    let tensor = Tensor::<NdArray, 2>::from_data(data.convert::<f32>(), &Default::default());
    let tensor = tensor.reshape([1, 28, 28]);

    // normalize: make between [0,1] and make the mean =  0 and std = 1
    // values mean=0.1307,std=0.3081 were copied from Pytorch Mist Example
    // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
    let tensor = ((tensor / 255) - 0.1307) / 0.3081;
    let tensor = if !transforms.is_empty() {
        mangle_image_batch(transforms, tensor)
    } else {
        tensor
    };

    MnistItemPrepared {
        image: tensor,
        label: item.label,
    }
}

/// Mange the image by applying small random transformations to augment the dataset.
///
/// * `images` - The images with shape [batch size, height, width]
///
/// ## Return
///
/// The transformed images tensor with shape [batch size, height, width]
fn mangle_image_batch<B: Backend>(transforms: &[Transform], images: Tensor<B, 3>) -> Tensor<B, 3> {
    let mut rng = rand::rng();

    let transforms = transforms.iter().map(|transform| match transform {
        Transform::Translate => {
            Transform2D::translation(rng.random_range(-0.2..0.2), rng.random_range(-0.2..0.2))
        }
        Transform::Shear => Transform2D::shear(
            rng.random_range(-0.6..0.6),
            rng.random_range(-0.6..0.6),
            0.0,
            0.0,
        ),
        Transform::Scale => Transform2D::scale(
            rng.random_range(0.6..1.5),
            rng.random_range(0.6..1.5),
            0.0,
            0.0,
        ),
        Transform::Rotation => {
            Transform2D::rotation(rng.random_range(-FRAC_PI_4..FRAC_PI_4), 0.0, 0.0)
        }
    });

    Transform2D::composed(transforms)
        .transform(images.unsqueeze_dim::<4>(1))
        .squeeze_dims::<3>(&[1])
}
