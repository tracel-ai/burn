use super::{dataset::TextGenerationItem, tokenizer::Tokenizer};
use burn::{
    data::dataloader::batcher::Batcher,
    nn::attention::generate_padding_mask,
    tensor::{backend::Backend, BoolTensor, Tensor},
};
use std::sync::Arc;

#[derive(new)]
pub struct TextGenerationBatcher {
    tokenizer: Arc<dyn Tokenizer>,
    vocab_size: usize,
    pad_token: usize,
    max_seq_lenght: usize,
}

#[derive(Debug, Clone, new)]
pub struct TextGenerationBatch<B: Backend> {
    pub tokens: Tensor<B::IntegerBackend, 2>,
    pub mask_pad: BoolTensor<B, 2>,
}

#[derive(Debug, Clone, new)]
pub struct TrainingTextGenerationBatch<B: Backend> {
    pub tokens_inputs: Tensor<B::IntegerBackend, 2>,
    pub targets: Tensor<B, 2>,
    pub mask_pad: BoolTensor<B, 2>,
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
            Some(self.max_seq_lenght),
            B::Device::default(),
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

        let inputs = item.tokens.index([0..batch_size, 0..seq_length - 1]);
        let targets = item.tokens.index([0..batch_size, 1..seq_length]);
        let mask_pad = item.mask_pad.index([0..batch_size, 0..seq_length - 1]);

        let seq_length = seq_length - 1;

        let targets = targets
            .reshape([batch_size * seq_length])
            .to_data()
            .value
            .iter()
            .map(|index| match *index as usize == self.pad_token {
                true => Tensor::<B, 2>::zeros([1, self.vocab_size]),
                false => Tensor::<B, 2>::one_hot(*index as usize, self.vocab_size),
            })
            .collect();

        let targets = Tensor::cat(targets, 0);

        TrainingTextGenerationBatch::new(inputs, targets, mask_pad)
    }
}
