use core::panic;
use std::marker::PhantomData;

use burn::{
    backend::VisionBackend,
    tensor::{ops::InterpolateMode, TensorPrimitive},
    vision::{
        create_structuring_element, KernelShape, MorphOptions, Morphology, Size, Transform2D
    },
};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
};
use rand::Rng;

use crate::show::{BatchDisplayOpts, ImageDimOrder, TensorDisplayOptions, save_as_img};

#[derive(Clone, Debug, Default)]
pub struct MnistBatcher<B: VisionBackend> {
    _p: PhantomData<B>,
}

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

struct PreparedMnistItem<B: Backend> {
    image: Tensor<B, 2>,
    label: B::IntElem,
}

impl<B: Backend, VB: VisionBackend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher<VB> {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        // concatenate into batches
        let (images, labels) = items
            .into_iter()
            .map(|item| {
                let device = &Device::<VB>::default();
                let image = Tensor::<VB, 2>::from_data(TensorData::from(item.image), device);
                let image = image.reshape([1, 28, 28]);
                let label = Tensor::<VB, 1, Int>::from_data(TensorData::from([item.label]), device);
                (image, label)
            })
            .unzip();

        let images: Tensor<VB, 3> = Tensor::cat(images, 0);
        let labels = Tensor::cat(labels, 0);

        // Random scaling, transformations etc
        let images = mangle_image_batch::<VB>(images);
        // normalize: make between [0,1] and make the mean =  0 and std = 1
        // values mean=0.1307,std=0.3081 were copied from Pytorch Mist Example
        // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
        let images = ((images / 255) - 0.1307) / 0.3081;

        let images = Tensor::<B, 3>::from_data(images.into_data(), device);
        let labels = Tensor::<B, 1, Int>::from_data(labels.into_data(), device);

        MnistBatch {
            images,
            targets: labels,
        }
    }
}

use std::time::Instant;


/// Prepares items for training
///
/// Preparation and augmentation is done on one backend, and the results are on another backend.
/// This is because we train on an AutodiffBackend, but we don't need autodiff to do vision ops.

/// Mange the image by applying small random transformations to augment the dataset.
///
/// * `images` - The images with shape [batch size, height, width]
///
/// ## Return
///
/// The transformed images tensor with shape [batch size, height, width]
fn mangle_image_batch<B: VisionBackend>(images: Tensor<B, 3>) -> Tensor<B, 3> {
    let mut rng = rand::rng();

    let [n, height, width] = images.shape().dims();

    // Resample
    let shear = Transform2D::shear(
        rng.random_range(-0.5..0.5),
        rng.random_range(-0.5..0.5),
        0.0,
        0.0,
    );
    let rotation = Transform2D::rotation(rng.random_range(-0.5..0.5), 0.0, 0.0);
    let scale = Transform2D::scale(
        rng.random_range(0.6..1.5),
        rng.random_range(0.6..1.5),
        0.0,
        0.0,
    );
    let translate =
        Transform2D::translation(rng.random_range(-0.2..0.2), rng.random_range(-0.2..0.2));
    
    let t0 = Instant::now();
    let transform = Transform2D::identity()
        .mul(translate)
        .mul(shear)
        .mul(scale)
        .mul(rotation);
    eprintln!("Transform creation: {:?}", t0.elapsed());

    let t1 = Instant::now();
    let grid = transform.mapping(n, [height, width]);
    eprintln!("Mapping: {:?}", t1.elapsed());

    let t2 = Instant::now();
    let images = images.reshape([n, 1, height, width]);
    eprintln!("Reshape before grid sample: {:?}", t2.elapsed());

    let t3 = Instant::now();
    let images = images.grid_sample_2d(grid, InterpolateMode::Bilinear);
    eprintln!("grid_sample_2d: {:?}", t3.elapsed());

    let t4 = Instant::now();
    let images = images.reshape([n, height, width]);
    eprintln!("Reshape after grid sample: {:?}", t4.elapsed());

    images
}
