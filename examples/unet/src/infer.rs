use std::fs;
use std::path::Path;

use burn::config::Config;
use burn::module::Module;
use burn::prelude::Backend;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::Shape;
use burn::tensor::Tensor;
use burn::tensor::TensorData;
use image::DynamicImage;
use image::ImageFormat;

use crate::brain_tumor_data::save_image;
use crate::brain_tumor_data::HEIGHT;
use crate::brain_tumor_data::WIDTH;
use crate::training::UNetTrainingConfig;
use crate::unet_model::{UNet, UNetRecord};

pub fn infer<B: Backend>(
    artifact_dir: &Path,
    infer_dir: &Path,
    device: &B::Device,
    source_image: &Path,
) {
    let config = UNetTrainingConfig::load(artifact_dir.join("config.json"))
        .expect("Config should exist for the model");
    let model_record: UNetRecord<B> = CompactRecorder::new()
        .load(artifact_dir.join("UNet").to_path_buf(), device)
        .expect("Trained model should exist; run train first");

    let unet: UNet<B> = config.model.init(device).load_record(model_record);

    // check the input image has the proper dimensions
    let dyn_image: DynamicImage = image::open(source_image).unwrap();
    if dyn_image.width() != WIDTH as u32 || dyn_image.height() != HEIGHT as u32 {
        panic!("Inference image does not have the proper width and height.")
    }

    // transform the image to burn tensor and run the forward method;
    let data = dyn_image.into_rgb8().into_raw();
    let source_tensor = Tensor::<B, 3>::from_data(
        TensorData::new(data, Shape::new([HEIGHT, WIDTH, 3])).convert::<B::FloatElem>(),
        device,
    )
    .swap_dims(0, 1)
    .swap_dims(0, 2)
    .div_scalar(255.0); // normalize range to [0.0, 1.0]

    let inferred_tensor: Tensor<B, 4> = unet.forward(source_tensor.reshape([1, 3, HEIGHT, WIDTH]));

    // for now, we'll assume that all input images and output images are png formatted.
    let inferred_image_name = source_image
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("default_name");

    let _ = fs::create_dir(infer_dir);
    let _ = save_image(
        inferred_tensor.clamp(0.0, 1.0).round().mul_scalar(255.0), // maps 0.0 -> black, 1.0 -> white
        Path::new(
            infer_dir
                .join(format!("target_segmentation_{}.png", inferred_image_name))
                .as_path(),
        ),
        ImageFormat::Png,
    )
    .ok();
}
