use super::{dataset::TextGenerationItem, tokenizer::Tokenizer};
use burn::{
    data::dataloader::batcher::Batcher,
    nn::attention::generate_padding_mask,
    tensor::{backend::Backend, BoolTensor, Tensor},
};
use std::sync::Arc;

#[derive(new)]
pub struct TextGenerationBatcher<B: Backend> {
    tokenizer: Arc<dyn Tokenizer>,
    device: B::Device,
    max_seq_lenght: usize,
}

#[derive(Debug, Clone, new)]
pub struct TextGenerationBatch<B: Backend> {
    pub tokens: Tensor<B::IntegerBackend, 2>,
    pub mask_pad: BoolTensor<B, 2>,
}

impl<B: Backend> Batcher<TextGenerationItem, TextGenerationBatch<B>> for TextGenerationBatcher<B> {
    fn batch(&self, items: Vec<TextGenerationItem>) -> TextGenerationBatch<B> {
        let mut tokens_list = Vec::with_capacity(items.len());

        for item in items {
            tokens_list.push(self.tokenizer.encode(&item.text, true));
        }

        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens_list,
            Some(self.max_seq_lenght),
            B::Device::default(),
        );

        TextGenerationBatch {
            tokens: mask.tensor.to_device(self.device).detach(),
            mask_pad: mask.mask.to_device(self.device),
        }
    }
}
