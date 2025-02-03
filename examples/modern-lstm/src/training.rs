use crate::dataset::{
    SequenceBatcher, SequenceDataset, NOISE_LEVEL, NUM_SEQUENCES, RANDOM_SEED, SEQ_LENGTH,
};
use crate::model::{LstmNetwork, LstmNetworkConfig};
use burn::{
    data::dataloader::DataLoaderBuilder,
    module::AutodiffModule,
    nn::loss::{MseLoss, Reduction::Mean},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
};

#[derive(Config)]
pub struct TrainingConfig {
    pub model: LstmNetworkConfig,
    pub optimizer: AdamConfig,

    #[config(default = 30)]
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 2)]
    pub num_workers: usize,
    #[config(default = 1e-3)]
    pub lr: f64,
}

// Create the directory to save the model and model config
fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);

    // Save training config
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");
    B::seed(RANDOM_SEED);

    // Create the model and optimizer
    let mut model = config.model.init::<B>(&device);
    let mut optim = config.optimizer.init::<B, LstmNetwork<B>>();

    // Create the batcher
    let batcher_train = SequenceBatcher::<B>::new(device.clone());
    let batcher_valid = SequenceBatcher::<B::InnerBackend>::new(device.clone());

    // Create the dataloaders
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(RANDOM_SEED)
        .num_workers(config.num_workers)
        .build(SequenceDataset::new(NUM_SEQUENCES, SEQ_LENGTH, NOISE_LEVEL));

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(RANDOM_SEED)
        .num_workers(config.num_workers)
        // 20% size of training
        .build(SequenceDataset::new(
            NUM_SEQUENCES / 5,
            SEQ_LENGTH,
            NOISE_LEVEL,
        ));

    let train_num_items = dataloader_train.num_items();
    let valid_num_items = dataloader_valid.num_items();

    println!("Starting training...");
    // Iterate over our training for X epochs
    for epoch in 1..config.num_epochs + 1 {
        // Initialize the training and validation metrics at the start of each epoch
        let mut train_losses = vec![];
        let mut train_loss = 0.0;
        let mut valid_losses = vec![];
        let mut valid_loss = 0.0;

        // Implement our training loop
        for batch in dataloader_train.iter() {
            let output = model.forward(batch.sequences, None);
            let loss = MseLoss::new().forward(output, batch.targets.clone(), Mean);
            train_loss += loss.clone().into_scalar().elem::<f32>() * batch.targets.dims()[0] as f32;

            // Gradients for the current backward pass
            let grads = loss.backward();
            // Gradients linked to each parameter of the model
            let grads = GradientsParams::from_grads(grads, &model);
            // Update the model using the optimizer
            model = optim.step(config.lr, model, grads);
        }

        // The averaged train loss per epoch
        let avg_train_loss = train_loss / train_num_items as f32;
        train_losses.push(avg_train_loss);

        // Get the model without autodiff
        let valid_model = model.valid();

        // Implement our validation loop
        for batch in dataloader_valid.iter() {
            let output = valid_model.forward(batch.sequences, None);
            let loss = MseLoss::new().forward(output, batch.targets.clone(), Mean);
            valid_loss += loss.clone().into_scalar().elem::<f32>() * batch.targets.dims()[0] as f32;
        }
        // The averaged train loss per epoch
        let avg_valid_loss = valid_loss / valid_num_items as f32;
        valid_losses.push(avg_valid_loss);

        // Display the averaged training and validataion metrics every 10 epochs
        if (epoch + 1) % 5 == 0 {
            println!(
                "Epoch {}/{}, Avg Loss {:.4}, Avg Val Loss: {:.4}",
                epoch + 1,
                config.num_epochs,
                avg_train_loss,
                avg_valid_loss,
            );
        }
    }

    // Save the trained model
    model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
