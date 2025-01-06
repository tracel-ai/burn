use clap::{Parser, Subcommand};

/// A CLI for generative and adversarial network
#[derive(Parser, Debug)]
#[command(version, author, about, long_about=None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Train a model
    Train {
        /// Path to save trained model
        #[arg(long)]
        artifact_dir: String,

        /// Number of epochs of training
        #[arg(long, default_value = "200")]
        num_epochs: usize,

        /// Number of training steps for discriminator before generator is trained per iteration
        #[arg(long, default_value = "5")]
        num_critic: usize,

        /// Size of the batches
        #[arg(long, default_value = "256")]
        batch_size: usize,

        /// Number of cpu threads to use during batch generation
        #[arg(long, default_value = "8")]
        num_workers: usize,

        /// Random seed
        #[arg(long, default_value = "5")]
        seed: u64,

        /// Learning rate
        #[arg(long, default_value = "0.00005")]
        lr: f64,

        /// Dimensionality of the latent space
        #[arg(long, default_value = "100")]
        latent_dim: usize,

        /// Size of each image dimension
        #[arg(long, default_value = "28")]
        image_size: usize,

        /// Number of image channels
        #[arg(long, default_value = "1")]
        channels: usize,

        /// Lower and upper clip value for disc. weights
        #[arg(long, default_value = "0.01")]
        clip_value: f32,

        /// Save images every sample_interval batches
        #[arg(long, default_value = "400")]
        sample_interval: usize,
    },

    // Generate images with the model
    Generate {
        // Path to the trained model
        #[arg(long)]
        artifact_dir: String,
    },
}
