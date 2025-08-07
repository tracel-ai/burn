use std::marker::PhantomData;

use burn::{
    backend::VisionBackend,
    tensor::TensorPrimitive,
    vision::{
        KernelShape, MorphOptions, Morphology, Size, Transform2D, create_structuring_element,
    },
};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
};
use rand::Rng;

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
        // Prepare items
        let items = items
            .iter()
            .map(|item| prepare_item::<VB>(item))
            .collect::<Vec<PreparedMnistItem<VB>>>();

        // concatenate into batches
        let (images, labels) = items
            .into_iter()
            .map(|item| {
                let image = item.image.reshape([1, 28, 28]);
                let label = Tensor::<VB, 1, Int>::from_data(
                    TensorData::from([item.label]),
                    &Default::default(),
                );
                (image, label)
            })
            .unzip();

        let images = Tensor::cat(images, 0);
        let labels = Tensor::cat(labels, 0);

        let images = Tensor::<B, 3>::from_data(images.into_data(), device);
        let labels = Tensor::<B, 1, Int>::from_data(labels.into_data(), device);

        MnistBatch {
            images,
            targets: labels,
        }
    }
}

/// Prepares items for training
///
/// Preparation and augmentation is done on one backend, and the results are on another backend.
/// This is because we train on an AutodiffBackend, but we don't need autodiff to do vision ops.
fn prepare_item<VB: VisionBackend>(item: &MnistItem) -> PreparedMnistItem<VB> {
    let data = TensorData::from(item.image);
    let image =
        Tensor::<VB, 2>::from_data(data.convert::<VB::FloatElem>(), &Device::<VB>::default());
    // Random scaling, transformations etc
    let image = mangle_image::<VB>(image);
    // normalize: make between [0,1] and make the mean =  0 and std = 1
    // values mean=0.1307,std=0.3081 were copied from Pytorch Mist Example
    // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
    let image = ((image / 255) - 0.1307) / 0.3081;

    // To training backend
    let label = (item.label as i64).elem::<VB::IntElem>();

    PreparedMnistItem { image, label }
}

/// Mange the image by applying small random transformations to augment the dataset.
///
/// * `tensor` - The image with shape [width, height]
///
/// ## Return
///
/// The transformed image tensor
fn mangle_image<B: VisionBackend>(image: Tensor<B, 2>) -> Tensor<B, 2> {
    let mut rng = rand::rng();
    // TODO implement resample for Tensor, no prim
    let image = image.reshape([1, 28, 28]).into_primitive().tensor();

    // Resample
    let shear = Transform2D::shear(
        rng.random_range(-0.5..0.5),
        rng.random_range(-0.5..0.5),
        14.0,
        14.0,
    );
    let rotation = Transform2D::rotation(rng.random_range(-0.5..0.5), 14.0, 14.0);
    let scale = Transform2D::scale(
        rng.random_range(0.6..1.5),
        rng.random_range(0.6..1.5),
        14.0,
        14.0,
    );
    let translate =
        Transform2D::translation(rng.random_range(-8.0..8.0), rng.random_range(-8.0..8.0));
    let transform = translate.mul(shear).mul(scale).mul(rotation);
    let image = B::float_resample(image, transform, 0.0);

    let image = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(image));

    // Erode / dialate image
    // let radius = rng.random_range(0..5);
    // let kernel = create_structuring_element::<B>(
    //     KernelShape::Ellipse,
    //     Size::new(radius, radius),
    //     None,
    //     &Default::default(),
    // );

    // let image = if rng.random_bool(0.5) {
    //     // Erode
    //     image.erode(kernel, MorphOptions::default())
    // } else {
    //     // Dialate
    //     image.dilate(kernel, MorphOptions::default())
    // };

    image.reshape([28, 28])
}
