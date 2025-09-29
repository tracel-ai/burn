// The module defines two structs TextClassificationTrainingBatch and TextClassificationInferenceBatch
// to handle batches of data during training and inference respectively. The TextClassificationBatcher
// struct is implemented for creating these batches. It is parameterized on the type B: Backend to
// support different computation backends (e.g., CPU, CUDA).

// Two implementations of the Batcher trait are provided for TextClassificationBatcher, one for creating
// training batches and one for creating inference batches. In each implementation, the batch function is
// defined to convert a vector of items into a batch. For training, the items are instances of
// TextClassificationItem and include both the text and the corresponding label.
// For inference, the items are simply strings without labels. The function tokenizes the text,
// generates a padding mask, and returns a batch object.

use super::{dataset::TextClassificationItem, tokenizer::Tokenizer};
use burn::{
    data::dataloader::batcher::Batcher,
    nn::attention::{SeqLengthOption, generate_padding_mask},
    prelude::*,
};
use std::sync::Arc;

/// Struct for batching text classification items
#[derive(Clone, new)]
pub struct TextClassificationBatcher {
    tokenizer: Arc<dyn Tokenizer>, // Tokenizer for converting text to token IDs
    seq_length: SeqLengthOption,   // Sequence length option for tokenized text
}

/// Struct for training batch in text classification task
#[derive(Debug, Clone, new)]
pub struct TextClassificationTrainingBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,    // Tokenized text
    pub labels: Tensor<B, 1, Int>,    // Labels of the text
    pub mask_pad: Tensor<B, 2, Bool>, // Padding mask for the tokenized text
}

/// Struct for inference batch in text classification task
#[derive(Debug, Clone, new)]
pub struct TextClassificationInferenceBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,    // Tokenized text
    pub mask_pad: Tensor<B, 2, Bool>, // Padding mask for the tokenized text
}

/// Implement Batcher trait for TextClassificationBatcher struct for training
impl<B: Backend> Batcher<B, TextClassificationItem, TextClassificationTrainingBatch<B>>
    for TextClassificationBatcher
{
    /// Batches a vector of text classification items into a training batch
    fn batch(
        &self,
        items: Vec<TextClassificationItem>,
        device: &B::Device,
    ) -> TextClassificationTrainingBatch<B> {
        let mut tokens_list = Vec::with_capacity(items.len());
        let mut labels_list = Vec::with_capacity(items.len());

        // Tokenize text and create label tensor for each item
        for item in items {
            tokens_list.push(self.tokenizer.encode(&item.text));
            labels_list.push(Tensor::from_data(
                TensorData::from([(item.label as i64).elem::<B::IntElem>()]),
                device,
            ));
        }

        // Generate padding mask for tokenized text
        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens_list,
            self.seq_length,
            device,
        );

        // Create and return training batch
        TextClassificationTrainingBatch {
            tokens: mask.tensor,
            labels: Tensor::cat(labels_list, 0),
            mask_pad: mask.mask,
        }
    }
}

/// Implement Batcher trait for TextClassificationBatcher struct for inference
impl<B: Backend> Batcher<B, String, TextClassificationInferenceBatch<B>>
    for TextClassificationBatcher
{
    /// Batches a vector of strings into an inference batch
    fn batch(&self, items: Vec<String>, device: &B::Device) -> TextClassificationInferenceBatch<B> {
        let mut tokens_list = Vec::with_capacity(items.len());

        // Tokenize each string
        for item in items {
            tokens_list.push(self.tokenizer.encode(&item));
        }

        // Generate padding mask for tokenized text
        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens_list,
            self.seq_length,
            device,
        );

        // Create and return inference batch
        TextClassificationInferenceBatch {
            tokens: mask.tensor.to_device(device),
            mask_pad: mask.mask.to_device(device),
        }
    }
}
