use crate::{
    model::ModelConfig,
    training::{save_image, TrainingConfig},
};
use burn::{
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::Distribution,
};
use std::path::Path;

pub fn generate<B: Backend>(artifact_dir: &str, device: B::Device) {
    // Loading model
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/generator").into(), &device)
        .expect("Trained model should exist; run train first");
    let config_model = ModelConfig::new(config.latent_dim, config.image_size, config.channels);
    let (mut generator, _) = config_model.init::<B>(&device);
    generator = generator.load_record(record);

    // Get a batch of noise
    let noise = Tensor::<B, 2>::random(
        [config.batch_size, config.latent_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let fake_images = generator.forward(noise); // [batch_size, channesl*height*width]
    let fake_images = fake_images.reshape([
        config.batch_size,
        config.channels,
        config.image_size as usize,
        config.image_size as usize,
    ]);
    // [B, C, H, W] to [B, H, C, W] to [B, H, W, C]
    let fake_images = fake_images.swap_dims(2, 1).swap_dims(3, 2).slice([0..25]);
    // Normalize the images. The Rgb32 images should be in range 0.0-1.0
    let fake_images = (fake_images.clone() - fake_images.clone().min().reshape([1,1,1,1]))
        / (fake_images.clone().max().reshape([1, 1, 1, 1])
           - fake_images.clone().min().reshape([1, 1, 1, 1]));
    // Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer, refer to pytorch save_image source
    let fake_images = (fake_images + 0.5 / 255.0).clamp(0.0, 1.0);
    // Save images in current directory
    let path = format!("fake_image.png");
    let path = Path::new(&path);
    save_image::<B, _>(fake_images, 5, path).unwrap();
}