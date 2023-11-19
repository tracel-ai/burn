use super::{dataset::TextGenerationItem, tokenizer::Tokenizer};
use burn::{
    data::dataloader::batcher::Batcher,
    nn::attention::generate_padding_mask,
    tensor::{backend::Backend, Bool, Int, Tensor},
};
use std::sync::Arc;

#[derive(new)]
pub struct TextGenerationBatcher {
    tokenizer: Arc<dyn Tokenizer>,
    max_seq_length: usize,
}

#[derive(Debug, Clone, new)]
pub struct TextGenerationBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub mask_pad: Tensor<B, 2, Bool>,
}

#[derive(Debug, Clone, new)]
pub struct TrainingTextGenerationBatch<B: Backend> {
    pub tokens_inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
    pub mask_pad: Tensor<B, 2, Bool>,
}

impl<B: Backend> Batcher<TextGenerationItem, TextGenerationBatch<B>> for TextGenerationBatcher {
    fn batch(&self, items: Vec<TextGenerationItem>) -> TextGenerationBatch<B> {
        let mut tokens_list = Vec::with_capacity(items.len());

        for item in items {
            tokens_list.push(self.tokenizer.encode(&item.text, true));
        }

        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens_list,
            Some(self.max_seq_length),
            &B::Device::default(),
        );

        TextGenerationBatch {
            tokens: mask.tensor,
            mask_pad: mask.mask,
        }
    }
}

impl<B: Backend> Batcher<TextGenerationItem, TrainingTextGenerationBatch<B>>
    for TextGenerationBatcher
{
    fn batch(&self, items: Vec<TextGenerationItem>) -> TrainingTextGenerationBatch<B> {
        let item: TextGenerationBatch<B> = self.batch(items);
        let [batch_size, seq_length] = item.tokens.dims();

        let inputs = item
            .tokens
            .clone()
            .slice([0..batch_size, 0..seq_length - 1]);
        let targets = item.tokens.slice([0..batch_size, 1..seq_length]);
        let mask_pad = item.mask_pad.slice([0..batch_size, 0..seq_length - 1]);

        TrainingTextGenerationBatch::new(inputs, targets, mask_pad)
    }
}
