use super::{dataset::TextGenerationItem, tokenizer::Tokenizer};
use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, BoolTensor, Data, Shape, Tensor},
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

        let (tokens, mask_pad) =
            pad_tokens::<B>(self.tokenizer.pad_token(), tokens_list, self.max_seq_lenght);

        TextGenerationBatch {
            tokens: tokens.to_device(self.device).detach(),
            mask_pad: mask_pad.to_device(self.device),
        }
    }
}

pub fn pad_tokens<B: Backend>(
    pad_token: usize,
    tokens_list: Vec<Vec<usize>>,
    max_seq_lenght: usize,
) -> (Tensor<B::IntegerBackend, 2>, BoolTensor<B, 2>) {
    let mut max_size = 0;
    let batch_size = tokens_list.len();

    for tokens in tokens_list.iter() {
        if tokens.len() > max_size {
            max_size = tokens.len();
        }
        if tokens.len() >= max_seq_lenght {
            max_size = max_seq_lenght;
            break;
        }
    }

    let mut tensor = Tensor::zeros([batch_size, max_size]);
    tensor = tensor.add_scalar(pad_token as i64);

    for (index, tokens) in tokens_list.into_iter().enumerate() {
        let mut seq_length = tokens.len();
        let mut tokens = tokens;
        if seq_length > max_seq_lenght {
            seq_length = max_seq_lenght;
            let _ = tokens.split_off(seq_length);
        }
        tensor = tensor.index_assign(
            [index..index + 1, 0..tokens.len()],
            &Tensor::from_data(Data::new(
                tokens.into_iter().map(|e| e as i64).collect(),
                Shape::new([1, seq_length]),
            )),
        );
    }

    let mask_pad = BoolTensor::from_int_backend(tensor.equal_scalar(pad_token as i64));

    (tensor, mask_pad)
}
