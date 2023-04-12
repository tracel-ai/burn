use super::{dataset::TextClassificationItem, tokenizer::Tokenizer};
use burn::{
    data::dataloader::batcher::Batcher,
    nn::attention::generate_padding_mask,
    tensor::{backend::Backend, Bool, Data, ElementConversion, Int, Tensor},
};
use std::sync::Arc;

#[derive(new)]
pub struct TextClassificationBatcher<B: Backend> {
    tokenizer: Arc<dyn Tokenizer>,
    device: B::Device,
    max_seq_lenght: usize,
}

#[derive(Debug, Clone, new)]
pub struct TextClassificationTrainingBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub labels: Tensor<B, 1, Int>,
    pub mask_pad: Tensor<B, 2, Bool>,
}

#[derive(Debug, Clone, new)]
pub struct TextClassificationInferenceBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub mask_pad: Tensor<B, 2, Bool>,
}

impl<B: Backend> Batcher<TextClassificationItem, TextClassificationTrainingBatch<B>>
    for TextClassificationBatcher<B>
{
    fn batch(&self, items: Vec<TextClassificationItem>) -> TextClassificationTrainingBatch<B> {
        let mut tokens_list = Vec::with_capacity(items.len());
        let mut labels_list = Vec::with_capacity(items.len());

        for item in items {
            tokens_list.push(self.tokenizer.encode(&item.text));
            labels_list.push(Tensor::from_data(Data::from([(item.label as i64).elem()])));
        }

        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens_list,
            Some(self.max_seq_lenght),
            &B::Device::default(),
        );

        TextClassificationTrainingBatch {
            tokens: mask.tensor.to_device(&self.device),
            labels: Tensor::cat(labels_list, 0).to_device(&self.device),
            mask_pad: mask.mask.to_device(&self.device),
        }
    }
}

impl<B: Backend> Batcher<String, TextClassificationInferenceBatch<B>>
    for TextClassificationBatcher<B>
{
    fn batch(&self, items: Vec<String>) -> TextClassificationInferenceBatch<B> {
        let mut tokens_list = Vec::with_capacity(items.len());

        for item in items {
            tokens_list.push(self.tokenizer.encode(&item));
        }

        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens_list,
            Some(self.max_seq_lenght),
            &B::Device::default(),
        );

        TextClassificationInferenceBatch {
            tokens: mask.tensor.to_device(&self.device),
            mask_pad: mask.mask.to_device(&self.device),
        }
    }
}
