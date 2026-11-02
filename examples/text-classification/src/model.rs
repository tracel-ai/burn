// This is a basic text classification model implemented in Rust using the Burn framework.
// It uses a Transformer as the base model and applies Linear and Embedding layers.
// The model is then trained using Cross-Entropy loss. It contains methods for model initialization
// (both with and without pre-trained weights), forward pass, inference, training, and validation.

use crate::data::{TextClassificationInferenceBatch, TextClassificationTrainingBatch};
use burn::{
    instrument::{Bounds, compute},
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        attention::SeqLengthOption,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    prelude::*,
    tensor::activation::softmax,
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};

// Define the model configuration
#[derive(Config, Debug)]
pub struct TextClassificationModelConfig {
    transformer: TransformerEncoderConfig,
    n_classes: usize,
    vocab_size: usize,
    seq_length: SeqLengthOption,
}

// Define the model structure
#[derive(Module, Debug)]
pub struct TextClassificationModel {
    transformer: TransformerEncoder,
    embedding_token: Embedding,
    embedding_pos: Embedding,
    output: Linear,
    n_classes: usize,
}

// Define functions for model initialization
impl TextClassificationModelConfig {
    /// Initializes a model with default weights
    pub fn init(&self, device: &Device) -> TextClassificationModel {
        let output = LinearConfig::new(self.transformer.d_model, self.n_classes).init(device);
        let transformer = self.transformer.init(device);
        let embedding_token =
            EmbeddingConfig::new(self.vocab_size, self.transformer.d_model).init(device);
        let max_seq_length = match self.seq_length {
            SeqLengthOption::Fixed(max) | SeqLengthOption::Max(max) => max,
            SeqLengthOption::NoMax => panic!(
                "Text classification requires a max sequence length because of the embedding strategy."
            ),
        };
        let embedding_pos =
            EmbeddingConfig::new(max_seq_length, self.transformer.d_model).init(device);

        TextClassificationModel {
            transformer,
            embedding_token,
            embedding_pos,
            output,
            n_classes: self.n_classes,
        }
    }
}

/// Define model behavior
impl TextClassificationModel {
    // Defines forward pass for training
    pub fn forward(&self, item: TextClassificationTrainingBatch) -> ClassificationOutput {
        // Get batch and sequence length, and the device
        let [batch_size, seq_length] = item.tokens.dims();
        assert_shape!(item.mask_pad, [batch_size, seq_length]);
        assert_shape!(item.labels, [batch_size]);
        let device = &self.embedding_token.devices()[0];

        // Move tensors to the correct device
        let tokens = item.tokens.to_device(device);
        let labels = item.labels.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        let embedding = self.embed(tokens);

        // Perform transformer encoding, calculate output and loss
        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(embedding).mask_pad(mask_pad));
        let output = self.output.forward(encoded);

        let output_classification = output
            .slice([0..batch_size, 0..1])
            .reshape([batch_size, self.n_classes]);

        let loss = CrossEntropyLossConfig::new()
            .init(&output_classification.device())
            .forward(output_classification.clone(), labels.clone());

        // Return the output and loss
        ClassificationOutput {
            loss,
            output: output_classification,
            targets: labels,
        }
    }

    /// Defines forward pass for inference
    #[compute(bounds = Self::bounds)]
    pub fn infer(&self, item: TextClassificationInferenceBatch) -> Tensor<2> {
        // Get batch and sequence length, and the device
        let [batch_size, seq_length] = item.tokens.dims();
        assert_shape!(item.mask_pad, [batch_size, seq_length]);
        let device = &self.embedding_token.devices()[0];

        // Move tensors to the correct device
        let tokens = item.tokens.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        let embedding = self.embed(tokens);

        // Perform transformer encoding, calculate output and apply softmax for prediction
        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(embedding).mask_pad(mask_pad));
        let output = self.output.forward(encoded);
        let output = output
            .slice([0..batch_size, 0..1])
            .reshape([batch_size, self.n_classes]);

        softmax(output, 1)
    }

    /// The floor for classifying `item` alone: read the tokens, the mask, the embedding rows they
    /// select and every other parameter once, write the class probabilities once, at two flops
    /// per multiply-add of the encoder and the head. Everything in between, every layer's
    /// activations and attention weights, is what a perfect fuser keeps out of memory.
    fn bounds(&self, item: &TextClassificationInferenceBatch) -> Bounds {
        let [batch_size, seq_length] = item.tokens.dims().map(|dim| dim as u64);
        let transformer = &self.transformer;
        let [d_model, d_ff, n_layers] =
            [transformer.d_model, transformer.d_ff, transformer.n_layers].map(|dim| dim as u64);
        let width = self.output.weight.val().dtype().size() as u64;
        let rows = batch_size * seq_length;
        let n_classes = self.n_classes as u64;

        // A row of the token table for every token, and one of the position table per position.
        let embeddings = (rows + seq_length) * d_model;
        let parameters = (transformer.num_params() + self.output.num_params()) as u64;
        let inputs = [item.tokens.dtype(), item.mask_pad.dtype()]
            .map(|dtype| rows * dtype.size() as u64)
            .iter()
            .sum::<u64>();

        // Per layer: the four attention projections, the scores and the context over every pair
        // of positions (the heads split the width, so their count does not change the sum), and
        // the feed-forward's two projections.
        let projections = 4 * rows * d_model * d_model;
        let attention = 2 * batch_size * seq_length * seq_length * d_model;
        let feed_forward = 2 * rows * d_model * d_ff;
        // The head projects every position and keeps the first; only that one is needed.
        let head = batch_size * d_model * n_classes;
        let multiply_adds = n_layers * (projections + attention + feed_forward) + head;

        Bounds {
            read: inputs + (embeddings + parameters) * width,
            write: batch_size * n_classes * width,
            flops: 2 * multiply_adds,
        }
    }

    /// Calculate token and position embeddings, and combine them.
    fn embed(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
        let [batch_size, seq_length] = tokens.dims();
        let index_positions = Tensor::arange(0..seq_length as i64, &tokens.device())
            .reshape([1, seq_length])
            .repeat_dim(0, batch_size);
        let embedding_positions = self.embed_positions(index_positions);
        let embedding_tokens = self.embed_tokens(tokens);
        (embedding_positions + embedding_tokens) / 2
    }

    // Both embeddings are an `Embedding`, which a measurement would merge into one component;
    // each is named for what it embeds.
    #[compute(label = "token embedding")]
    fn embed_tokens(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
        self.embedding_token.forward(tokens)
    }

    #[compute(label = "position embedding")]
    fn embed_positions(&self, positions: Tensor<2, Int>) -> Tensor<3> {
        self.embedding_pos.forward(positions)
    }

    pub fn reset_head(&mut self, num_classes: usize) {
        self.n_classes = num_classes;
        self.output = LinearConfig::new(self.transformer.d_model, self.n_classes)
            .init(&self.transformer.devices()[0]);
        self.output.weight = self.output.weight.clone().set_require_grad(true);
        self.output.bias = self.output.bias.clone().map(|b| b.set_require_grad(true));
    }
}

/// Define training step
impl TrainStep for TextClassificationModel {
    type Input = TextClassificationTrainingBatch;
    type Output = ClassificationOutput;

    fn step(&self, item: TextClassificationTrainingBatch) -> TrainOutput<ClassificationOutput> {
        // Run forward pass, calculate gradients and return them along with the output
        let item = self.forward(item);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

/// Define validation step
impl InferenceStep for TextClassificationModel {
    type Input = TextClassificationTrainingBatch;
    type Output = ClassificationOutput;

    fn step(&self, item: TextClassificationTrainingBatch) -> ClassificationOutput {
        // Run forward pass and return the output
        self.forward(item)
    }
}
