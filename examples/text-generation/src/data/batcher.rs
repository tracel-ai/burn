use super::{dataset::TextGenerationItem, tokenizer::Tokenizer};
use burn::{data::dataloader::batcher::Batcher, nn::attention::generate_padding_mask, prelude::*};
use std::sync::Arc;

#[derive(Clone, new)]
pub struct TextGenerationBatcher {
    tokenizer: Arc<dyn Tokenizer>,
    max_seq_length: usize,
}

#[derive(Debug, Clone, new)]
pub struct TextGenerationBatch {
    pub tokens: Tensor<2, Int>,
    pub mask_pad: Tensor<2, Bool>,
}

#[derive(Debug, Clone, new)]
pub struct TrainingTextGenerationBatch {
    pub tokens_inputs: Tensor<2, Int>,
    pub targets: Tensor<2, Int>,
    pub mask_pad: Tensor<2, Bool>,
}

impl Batcher<TextGenerationItem, TextGenerationBatch> for TextGenerationBatcher {
    fn batch(&self, items: Vec<TextGenerationItem>, device: &Device) -> TextGenerationBatch {
        let mut tokens_list = Vec::with_capacity(items.len());

        for item in items {
            tokens_list.push(self.tokenizer.encode(&item.text, true));
        }

        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens_list,
            Some(self.max_seq_length),
            device,
        );

        TextGenerationBatch {
            tokens: mask.tensor,
            mask_pad: mask.mask,
        }
    }
}

impl Batcher<TextGenerationItem, TrainingTextGenerationBatch> for TextGenerationBatcher {
    fn batch(
        &self,
        items: Vec<TextGenerationItem>,
        device: &Device,
    ) -> TrainingTextGenerationBatch {
        let item: TextGenerationBatch = self.batch(items, device);
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
