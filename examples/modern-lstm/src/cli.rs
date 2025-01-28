use clap::{Parser, Subcommand};

/// A CLI for long short-term memory network
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

        /// Size of the batches
        #[arg(long, default_value = "64")]
        batch_size: usize,

        /// Number of cpu threads to use during batch generation
        #[arg(long, default_value = "8")]
        num_workers: usize,

        /// Learning rate
        #[arg(long, default_value = "0.00005")]
        lr: f64,
    },

    /// Inference
    Infer {
        /// Path to the trained model
        #[arg(long)]
        artifact_dir: String,
    },
}
