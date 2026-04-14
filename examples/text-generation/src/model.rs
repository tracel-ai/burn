use crate::data::TrainingTextGenerationBatch;
use burn::{
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        attention::generate_autoregressive_mask,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    prelude::*,
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};

#[derive(Config, Debug)]
pub struct TextGenerationModelConfig {
    transformer: TransformerEncoderConfig,
    vocab_size: usize,
    pad_token: usize,
    max_seq_length: usize,
}

#[derive(Module, Debug)]
pub struct TextGenerationModel {
    transformer: TransformerEncoder,
    embedding_token: Embedding,
    embedding_pos: Embedding,
    output: Linear,
    vocab_size: usize,
    pad_token: usize,
    max_seq_length: usize,
}

impl TextGenerationModelConfig {
    pub fn init(&self, device: &Device) -> TextGenerationModel {
        let output = LinearConfig::new(self.transformer.d_model, self.vocab_size).init(device);
        let transformer = self.transformer.init(device);
        let embedding_token =
            EmbeddingConfig::new(self.vocab_size, self.transformer.d_model).init(device);
        let embedding_pos =
            EmbeddingConfig::new(self.max_seq_length, self.transformer.d_model).init(device);

        TextGenerationModel {
            transformer,
            embedding_token,
            embedding_pos,
            output,
            vocab_size: self.vocab_size,
            pad_token: self.pad_token,
            max_seq_length: self.max_seq_length,
        }
    }
}
impl TextGenerationModel {
    pub fn forward_training(&self, item: TrainingTextGenerationBatch) -> ClassificationOutput {
        let [batch_size, seq_length] = item.tokens_inputs.dims();
        let device = &self.devices()[0];

        let inputs = item.tokens_inputs.to_device(device);
        let targets = item.targets.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        let index_positions = Tensor::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat_dim(0, batch_size);

        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(inputs);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        let mask_attn = generate_autoregressive_mask(batch_size, seq_length, device);
        let encoded = self.transformer.forward(
            TransformerEncoderInput::new(embedding)
                .mask_pad(mask_pad)
                .mask_attn(mask_attn),
        );

        let output = self.output.forward(encoded);
        let output_flatten = output.reshape([batch_size * seq_length, self.vocab_size]);
        let targets_flatten = targets.reshape([batch_size * seq_length]);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![self.pad_token]))
            .init(&output_flatten.device());
        let loss = loss.forward(output_flatten.clone(), targets_flatten.clone());

        ClassificationOutput {
            loss,
            output: output_flatten,
            targets: targets_flatten,
        }
    }
}

impl TrainStep for TextGenerationModel {
    type Input = TrainingTextGenerationBatch;
    type Output = ClassificationOutput;

    fn step(&self, item: TrainingTextGenerationBatch) -> TrainOutput<ClassificationOutput> {
        let item = self.forward_training(item);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl InferenceStep for TextGenerationModel {
    type Input = TrainingTextGenerationBatch;
    type Output = ClassificationOutput;

    fn step(&self, item: TrainingTextGenerationBatch) -> ClassificationOutput {
        self.forward_training(item)
    }
}
