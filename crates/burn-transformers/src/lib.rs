pub mod cache;
pub mod models;
pub mod pretrained;
pub mod sampling;
pub mod tokenizer;
pub mod transformer;

#[cfg(test)]
mod tests {
    use burn::backend::Candle;
    pub type TestBackend = Candle<f32, i64>;

    pub type TestTensor<const D: usize> = burn::tensor::Tensor<TestBackend, D>;
}
