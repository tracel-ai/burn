use crate::dataset::MnistBatcher;
use crate::model::{Clip, ModelConfig};
use burn::optim::{GradientsParams, Optimizer, RmsPropConfig};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    prelude::*,
    record::CompactRecorder,
    tensor::{Distribution, backend::AutodiffBackend},
};
use image::{Rgb32FImage, RgbImage, buffer::ConvertBuffer, error::ImageResult};
use std::path::Path;

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: RmsPropConfig,

    #[config(default = 200)]
    pub num_epochs: usize,
    #[config(default = 512)]
    pub batch_size: usize,
    #[config(default = 8)]
    pub num_workers: usize,
    #[config(default = 5)]
    pub seed: u64,
    #[config(default = 3e-4)]
    pub lr: f64,

    /// Number of training steps for discriminator before generator is trained per iteration
    #[config(default = 5)]
    pub num_critic: usize,
    /// Lower and upper clip value for disc. weights
    #[config(default = 0.01)]
    pub clip_value: f32,
    /// Save a sample of images every `sample_interval` epochs
    #[config(default = 10)]
    pub sample_interval: usize,
}

// Create the directory to save the model and model config
fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

/// Save the generated images
// The images format is [B, H, W, C]
pub fn save_image<B: Backend, Q: AsRef<Path>>(
    images: Tensor<B, 4>,
    nrow: u32,
    path: Q,
) -> ImageResult<()> {
    let ncol = (images.dims()[0] as f32 / nrow as f32).ceil() as u32;

    let width = images.dims()[2] as u32;
    let height = images.dims()[1] as u32;

    // Supports both 1 and 3 channels image
    let channels = match images.dims()[3] {
        1 => 3,
        3 => 1,
        _ => panic!("Wrong channels number"),
    };

    let mut imgbuf = RgbImage::new(nrow * width, ncol * height);
    // Write images into a nrow*ncol grid layout
    for row in 0..nrow {
        for col in 0..ncol {
            let image: Tensor<B, 3> = images
                .clone()
                .slice((row * nrow + col) as usize..(row * nrow + col + 1) as usize)
                .squeeze_dim(0);
            // The Rgb32 should be in range 0.0-1.0
            let image = image.into_data().iter::<f32>().collect::<Vec<f32>>();
            // Supports both 1 and 3 channels image
            let image = image
                .into_iter()
                .flat_map(|n| std::iter::repeat_n(n, channels))
                .collect();

            let image = Rgb32FImage::from_vec(width, height, image).unwrap();
            let image: RgbImage = image.convert();
            for (x, y, pixel) in image.enumerate_pixels() {
                imgbuf.put_pixel(row * width + x, col * height + y, *pixel);
            }
        }
    }
    imgbuf.save(path)
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);

    // Create the Clip module mapper
    let mut clip = Clip {
        min: -config.clip_value,
        max: config.clip_value,
    };

    // Save training config
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");
    B::seed(&device, config.seed);

    // Create the model and optimizer
    let (mut generator, mut discriminator) = config.model.init::<B>(&device);
    let mut optimizer_g = config.optimizer.init();
    let mut optimizer_d = config.optimizer.init();

    // Create the dataset batcher
    let batcher_train = MnistBatcher::default();

    // Create the dataloaders
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());

    // Iterate over our training for X epochs
    for epoch in 0..config.num_epochs {
        // Implement our training loop
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            // Generate a batch of fake images from noise (standarded normal distribution)
            let noise = Tensor::<B, 2>::random(
                [config.batch_size, config.model.latent_dim],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            // datach: do not update gerenator, only discriminator is updated
            let fake_images = generator.forward(noise.clone()).detach(); // [batch_size, channels*height*width]
            let fake_images = fake_images.reshape([
                config.batch_size,
                config.model.channels,
                config.model.image_size,
                config.model.image_size,
            ]);
            // Adversarial loss
            let loss_d = -discriminator.forward(batch.images).mean()
                + discriminator.forward(fake_images.clone()).mean();

            // Gradients for the current backward pass
            let grads = loss_d.backward();
            // Gradients linked to each parameter of the discriminator
            let grads = GradientsParams::from_grads(grads, &discriminator);
            // Update the discriminator using the optimizer
            discriminator = optimizer_d.step(config.lr, discriminator, grads);
            // Clip parameters (weights) of discriminator
            discriminator = discriminator.map(&mut clip);

            // Train the generator every num_critic iterations
            if iteration % config.num_critic == 0 {
                // Generate a batch of images again without detaching
                let critic_fake_images = generator.forward(noise.clone());
                let critic_fake_images = critic_fake_images.reshape([
                    config.batch_size,
                    config.model.channels,
                    config.model.image_size,
                    config.model.image_size,
                ]);
                // Adversarial loss. Minimize it to make the fake images as truth
                let loss_g = -discriminator.forward(critic_fake_images).mean();

                let grads = loss_g.backward();
                let grads = GradientsParams::from_grads(grads, &generator);
                generator = optimizer_g.step(config.lr, generator, grads);

                // Print the progression
                let batch_num = (dataloader_train.num_items() as f32 / config.batch_size as f32)
                    .ceil() as usize;
                println!(
                    "[Epoch {}/{}] [Batch {}/{}] [D loss: {}] [G loss: {}]",
                    epoch + 1,
                    config.num_epochs,
                    iteration,
                    batch_num,
                    loss_d.into_scalar(),
                    loss_g.into_scalar()
                );
            }
            //  If at save interval => save the first 25 generated images
            if epoch % config.sample_interval == 0 && iteration == 0 {
                // [B, C, H, W] to [B, H, C, W] to [B, H, W, C]
                let fake_images = fake_images.swap_dims(2, 1).swap_dims(3, 2).slice(0..25);
                // Normalize the images. The Rgb32 images should be in range 0.0-1.0
                let fake_images = (fake_images.clone()
                    - fake_images.clone().min().reshape([1, 1, 1, 1]))
                    / (fake_images.clone().max().reshape([1, 1, 1, 1])
                        - fake_images.clone().min().reshape([1, 1, 1, 1]));
                // Add 0.5/255.0 to the images, refer to pytorch save_image source
                let fake_images = (fake_images + 0.5 / 255.0).clamp(0.0, 1.0);
                // Save images in artifact directory
                let path = format!("{artifact_dir}/image-{epoch}.png");
                save_image::<B, _>(fake_images, 5, path).unwrap();
            }
        }
    }

    // Save the trained models
    generator
        .save_file(format!("{artifact_dir}/generator"), &CompactRecorder::new())
        .expect("Generator should be saved successfully");
    discriminator
        .save_file(
            format!("{artifact_dir}/discriminator"),
            &CompactRecorder::new(),
        )
        .expect("Discriminator should be saved successfully");
}
