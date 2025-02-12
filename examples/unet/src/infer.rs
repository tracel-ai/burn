use std::fs;
use std::path::Path;

use burn::config::Config;
use burn::module::Module;
use burn::prelude::Backend;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::{Float, Tensor, TensorData};
use image::{DynamicImage, ImageFormat};

use crate::brain_tumor_data::save_image;
use crate::brain_tumor_data::source_dynamic_image_to_vector;
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
    // convert the predicted burn tensor back to an image and save the image to the artifact directory
    let source_vec: Vec<f32> = source_dynamic_image_to_vector(&image::open(source_image).unwrap());
    let source_array: Box<[f32]> = source_vec.into_boxed_slice();
    let source_data = TensorData::from(&*source_array).convert::<B::FloatElem>();
    let source_tensor: Tensor<B, 1, Float> = Tensor::from_data(source_data, device);
    let source_tensor: Tensor<B, 3, Float> =
        source_tensor.reshape([WIDTH, HEIGHT, 3]).swap_dims(0, 2);
    let inferred_tensor: Tensor<B, 4, Float> =
        unet.forward(source_tensor.reshape([1, 3, HEIGHT, WIDTH]));
    println!("{:}", inferred_tensor);

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
                .join(format!(
                    "predict_target_segmentation_{}.png",
                    inferred_image_name
                ))
                .as_path(),
        ),
        ImageFormat::Png,
    )
    .ok();
}
