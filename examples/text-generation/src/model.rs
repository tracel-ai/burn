use crate::data::TrainingTextGenerationBatch;
use burn::{
    config::Config,
    module::Module,
    nn::{
        attention::generate_autoregressive_mask,
        loss::CrossEntropyLoss,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Embedding, EmbeddingConfig, Linear, LinearConfig,
    },
    tensor::backend::{ADBackend, Backend},
    tensor::Tensor,
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Config)]
pub struct TextGenerationModelConfig {
    transformer: TransformerEncoderConfig,
    vocab_size: usize,
    pad_token: usize,
    max_seq_length: usize,
}

#[derive(Module, Debug)]
pub struct TextGenerationModel<B: Backend> {
    transformer: TransformerEncoder<B>,
    embedding_token: Embedding<B>,
    embedding_pos: Embedding<B>,
    output: Linear<B>,
    vocab_size: usize,
    pad_token: usize,
    max_seq_length: usize,
}

impl TextGenerationModelConfig {
    pub fn init<B: Backend>(&self) -> TextGenerationModel<B> {
        let output = LinearConfig::new(self.transformer.d_model, self.vocab_size).init();
        let transformer = self.transformer.init();
        let embedding_token =
            EmbeddingConfig::new(self.vocab_size, self.transformer.d_model).init();
        let embedding_pos =
            EmbeddingConfig::new(self.max_seq_length, self.transformer.d_model).init();

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
impl<B: Backend> TextGenerationModel<B> {
    pub fn forward_training(
        &self,
        item: TrainingTextGenerationBatch<B>,
    ) -> ClassificationOutput<B> {
        let [batch_size, seq_length] = item.tokens_inputs.dims();
        let device = &self.devices()[0];

        let inputs = item.tokens_inputs.to_device(device);
        let targets = item.targets.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        let index_positions = Tensor::arange_device(0..seq_length, device)
            .reshape([1, seq_length])
            .repeat(0, batch_size);

        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(inputs);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        let mask_attn = generate_autoregressive_mask::<B>(batch_size, seq_length, device);
        let encoded = self.transformer.forward(
            TransformerEncoderInput::new(embedding)
                .mask_pad(mask_pad)
                .mask_attn(mask_attn),
        );

        let output = self.output.forward(encoded);
        let output_flatten = output.reshape([batch_size * seq_length, self.vocab_size]);
        let targets_flatten = targets.reshape([batch_size * seq_length]);

        let loss = CrossEntropyLoss::new(Some(self.pad_token));
        let loss = loss.forward(output_flatten.clone(), targets_flatten.clone());

        ClassificationOutput {
            loss,
            output: output_flatten,
            targets: targets_flatten,
        }
    }
}

impl<B: ADBackend> TrainStep<TrainingTextGenerationBatch<B>, ClassificationOutput<B>>
    for TextGenerationModel<B>
{
    fn step(&self, item: TrainingTextGenerationBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_training(item);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<TrainingTextGenerationBatch<B>, ClassificationOutput<B>>
    for TextGenerationModel<B>
{
    fn step(&self, item: TrainingTextGenerationBatch<B>) -> ClassificationOutput<B> {
        self.forward_training(item)
    }
}
