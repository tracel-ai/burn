use burn::tensor::{backend::Backend, Int, Tensor};
use rand::{
    distr::{weighted::WeightedIndex, Distribution},
    rngs::StdRng,
    SeedableRng,
};

pub enum Sampler {
    TopP(TopP),
    Argmax,
}

impl Sampler {
    pub fn sample<B: Backend>(&mut self, logits: Tensor<B, 2>) -> Tensor<B, 2, Int> {
        match self {
            Self::TopP(s) => s.sample(logits),
            Self::Argmax => logits.argmax(1),
        }
    }
}

pub trait Sampling {
    fn sample<B: Backend>(&mut self, logits: Tensor<B, 2>) -> Tensor<B, 2, Int>;
}

/// Top-p sampling (nucleus sampling) selects the smallest set of tokens whose cumulative
/// probability mass exceed the threshold p.
pub struct TopP {
    /// Probability threshold for sampling.
    p: f64,
    /// RNG.
    rng: StdRng,
}

impl TopP {
    pub fn new(p: f64, seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        Self { p, rng }
    }
}

impl Sampling for TopP {
    fn sample<B: Backend>(&mut self, probs: Tensor<B, 2>) -> Tensor<B, 2, Int> {
        assert_eq!(
            probs.dims()[0],
            1,
            "Naive top-p sampling only supports single-batch tensors"
        );
        let (probs_sort, probs_idx) = probs.sort_descending_with_indices(1);

        // TODO: cumsum + Distribution::Multinomial support

        let mut probs_sort = probs_sort.to_data().iter::<f64>().collect::<Vec<_>>();

        let mut cumsum = 0.;
        probs_sort.iter_mut().for_each(|x| {
            if cumsum >= self.p {
                *x = 0.0;
            } else {
                cumsum += *x;
            }
        });

        let next_token_idx = WeightedIndex::new(probs_sort)
            .unwrap()
            .sample(&mut self.rng);

        probs_idx.slice([0..1, next_token_idx..next_token_idx + 1])
    }
}
