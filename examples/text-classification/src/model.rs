use crate::data::TextClassificationBatch;
use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Embedding, EmbeddingConfig, Linear, LinearConfig,
    },
    tensor::backend::{ADBackend, Backend},
    tensor::{loss::cross_entropy_with_logits, Tensor},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Config)]
pub struct TextClassificationModelConfig {
    transformer: TransformerEncoderConfig,
    n_classes: usize,
    vocab_size: usize,
}

#[derive(Module, Debug)]
pub struct TextClassificationModel<B: Backend> {
    transformer: Param<TransformerEncoder<B>>,
    embedding_token: Param<Embedding<B>>,
    embedding_pos: Param<Embedding<B>>,
    output: Param<Linear<B>>,
}

impl<B: Backend> TextClassificationModel<B> {
    pub fn new(config: &TextClassificationModelConfig) -> Self {
        let max_seq_length = 256;
        let config_embedding_token =
            EmbeddingConfig::new(config.vocab_size, config.transformer.d_model);
        let config_embedding_pos = EmbeddingConfig::new(max_seq_length, config.transformer.d_model);
        let config_output = LinearConfig::new(config.transformer.d_model, config.n_classes);

        let transformer = TransformerEncoder::new(&config.transformer);
        let embedding_token = Embedding::new(&config_embedding_token);
        let embedding_pos = Embedding::new(&config_embedding_pos);
        let output = Linear::new(&config_output);

        Self {
            transformer: Param::new(transformer),
            embedding_token: Param::new(embedding_token),
            embedding_pos: Param::new(embedding_pos),
            output: Param::new(output),
        }
    }

    pub fn forward(&self, item: TextClassificationBatch<B>) -> ClassificationOutput<B> {
        let [batch_size, seq_length] = item.tokens.dims();
        let pos = Tensor::<B, 1>::arange_device(0..seq_length, item.tokens.device())
            .reshape([1, seq_length])
            .repeat(0, batch_size);
        let x_pos = self.embedding_pos.forward(pos.detach());
        let x_tokens = self.embedding_token.forward(item.tokens.detach());
        let x = (x_pos + x_tokens) / 2;
        let x = self
            .transformer
            .forward(TransformerEncoderInput::new(x).mask_pad(item.mask_pad));
        let x = self.output.forward(x);

        let [batch_size, _seq_length, d_model] = x.dims();

        let x = x
            .index([0..batch_size, 0..1])
            .reshape([batch_size, d_model]);

        let loss = cross_entropy_with_logits(&x, &item.labels.clone().detach());

        ClassificationOutput {
            loss,
            output: x,
            targets: item.labels,
        }
    }
}

impl<B: ADBackend> TrainStep<TextClassificationBatch<B>, ClassificationOutput<B>, B::Gradients>
    for TextClassificationModel<B>
{
    fn step(
        &self,
        item: TextClassificationBatch<B>,
    ) -> TrainOutput<ClassificationOutput<B>, B::Gradients> {
        let item = self.forward(item);
        let grads = item.loss.backward();

        TrainOutput::new(grads, item)
    }
}

impl<B: Backend> ValidStep<TextClassificationBatch<B>, ClassificationOutput<B>>
    for TextClassificationModel<B>
{
    fn step(&self, item: TextClassificationBatch<B>) -> ClassificationOutput<B> {
        self.forward(item)
    }
}
