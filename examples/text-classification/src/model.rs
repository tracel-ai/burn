use crate::data::{TextClassificationInferenceBatch, TextClassificationTrainingBatch};
use burn::{
    config::Config,
    module::Module,
    nn::{
        loss::CrossEntropyLoss,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Embedding, EmbeddingConfig, Linear, LinearConfig,
    },
    tensor::backend::{ADBackend, Backend},
    tensor::{activation::softmax, Tensor},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Config)]
pub struct TextClassificationModelConfig {
    transformer: TransformerEncoderConfig,
    n_classes: usize,
    vocab_size: usize,
    max_seq_length: usize,
}

#[derive(Module, Debug)]
pub struct TextClassificationModel<B: Backend> {
    transformer: TransformerEncoder<B>,
    embedding_token: Embedding<B>,
    embedding_pos: Embedding<B>,
    output: Linear<B>,
    n_classes: usize,
    max_seq_length: usize,
}

impl TextClassificationModelConfig {
    pub fn init<B: Backend>(&self) -> TextClassificationModel<B> {
        let output = LinearConfig::new(self.transformer.d_model, self.n_classes).init();
        let transformer = self.transformer.init();
        let embedding_token =
            EmbeddingConfig::new(self.vocab_size, self.transformer.d_model).init();
        let embedding_pos =
            EmbeddingConfig::new(self.max_seq_length, self.transformer.d_model).init();

        TextClassificationModel {
            transformer,
            embedding_token,
            embedding_pos,
            output,
            n_classes: self.n_classes,
            max_seq_length: self.max_seq_length,
        }
    }
    pub fn init_with<B: Backend>(
        &self,
        record: TextClassificationModelRecord<B>,
    ) -> TextClassificationModel<B> {
        let output =
            LinearConfig::new(self.transformer.d_model, self.n_classes).init_with(record.output);
        let transformer = self.transformer.init_with(record.transformer);
        let embedding_token = EmbeddingConfig::new(self.vocab_size, self.transformer.d_model)
            .init_with(record.embedding_token);
        let embedding_pos = EmbeddingConfig::new(self.max_seq_length, self.transformer.d_model)
            .init_with(record.embedding_pos);

        TextClassificationModel {
            transformer,
            embedding_token,
            embedding_pos,
            output,
            n_classes: self.n_classes,
            max_seq_length: self.max_seq_length,
        }
    }
}

impl<B: Backend> TextClassificationModel<B> {
    pub fn forward(&self, item: TextClassificationTrainingBatch<B>) -> ClassificationOutput<B> {
        let [batch_size, seq_length] = item.tokens.dims();
        let device = &self.embedding_token.devices()[0];

        let tokens = item.tokens.to_device(device);
        let labels = item.labels.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        let index_positions = Tensor::arange_device(0..seq_length, device)
            .reshape([1, seq_length])
            .repeat(0, batch_size);
        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(tokens);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(embedding).mask_pad(mask_pad));
        let output = self.output.forward(encoded);

        let output_classification = output
            .index([0..batch_size, 0..1])
            .reshape([batch_size, self.n_classes]);

        let loss = CrossEntropyLoss::new(None);
        let loss = loss.forward(output_classification.clone(), labels.clone());

        ClassificationOutput {
            loss,
            output: output_classification,
            targets: labels,
        }
    }

    pub fn infer(&self, item: TextClassificationInferenceBatch<B>) -> Tensor<B, 2> {
        let [batch_size, seq_length] = item.tokens.dims();
        let device = &self.embedding_token.devices()[0];

        let tokens = item.tokens.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        let index_positions = Tensor::arange_device(0..seq_length, device)
            .reshape([1, seq_length])
            .repeat(0, batch_size);
        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(tokens);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(embedding).mask_pad(mask_pad));
        let output = self.output.forward(encoded);
        let output = output
            .index([0..batch_size, 0..1])
            .reshape([batch_size, self.n_classes]);

        softmax(output, 1)
    }
}

impl<B: ADBackend> TrainStep<TextClassificationTrainingBatch<B>, ClassificationOutput<B>>
    for TextClassificationModel<B>
{
    fn step(
        &self,
        item: TextClassificationTrainingBatch<B>,
    ) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward(item);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<TextClassificationTrainingBatch<B>, ClassificationOutput<B>>
    for TextClassificationModel<B>
{
    fn step(&self, item: TextClassificationTrainingBatch<B>) -> ClassificationOutput<B> {
        self.forward(item)
    }
}
