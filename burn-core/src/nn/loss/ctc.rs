#![allow(clippy::single_range_in_vec_init)]
use core::marker::PhantomData;

use burn_tensor::{backend::Backend, ElementConversion, Int, Tensor};
use half::f16;

use super::Reduction;

const NEG_INF: f16 = f16::NEG_INFINITY;

/// The Connectionist Temporal Classification loss.
#[derive(Clone, Debug)]
pub struct CTCLoss<B: Backend> {
    blank: usize,
    backend: PhantomData<B>,
}

impl<B: Backend> Default for CTCLoss<B> {
    fn default() -> Self {
        CTCLoss::new(0)
    }
}

impl<B: Backend> CTCLoss<B> {
    /// Create the criterion.
    pub fn new(blank: usize) -> Self {
        Self {
            blank,
            backend: PhantomData,
        }
    }

    /// Compute the criterion on the input tensor.
    ///
    /// # Parameters:
    ///
    /// - log_probs: The logarithmized probabilities of the outputs. Shape:
    ///   `[batch_size, input_length, num_classes]`
    /// - targets: It represent the concatenated  target sequences. Each
    ///   element in the target sequence is a class index. And the target
    ///   index cannot be blank. Shape: `[target_lengths_sum]`
    /// - input_lengths: It represent the lengths of the inputs. And the
    ///   lengths are specified for each sequence to achieve masking under
    ///   the assumption that sequences are padded to equal lengths. Shape:
    ///   `[batch_size]`
    /// - target_lengths:  It represent lengths of the targets. Shape:
    ///   `[batch_size]`
    /// - reduction: Specifies the reduction to apply to the output. None:
    ///   no reduction will be applied; Some(Reduction::Mean): the output
    ///   losses will be divided by the target lengths and then the mean
    ///   over the batch is taken; Some(Reduction::Sum): the output losses
    ///   will be summed.
    ///
    /// # Reference
    ///
    /// - [PyTorch implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/LossCTC.cpp)
    /// - [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
    pub fn forward(
        &self,
        log_probs: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
        input_lengths: Tensor<B, 1, Int>,
        target_lengths: Tensor<B, 1, Int>,
        reduction: Option<Reduction>,
    ) -> Tensor<B, 1> {
        Self::assertions(
            log_probs.clone(),
            targets.clone(),
            input_lengths.clone(),
            target_lengths.clone(),
        );

        let [batch_size, seq_length, _] = log_probs.dims();
        let max_target_length = target_lengths.clone().max().into_scalar().elem::<u32>() as usize;
        let target_with_blank_length = 2 * max_target_length + 1;

        let mut log_alphas =
            Tensor::<B, 3>::empty([batch_size, seq_length, target_with_blank_length]);
        log_alphas = log_alphas.slice_assign(
            [0..batch_size, 0..1, 0..target_with_blank_length],
            Tensor::<B, 3>::full([batch_size, 1, target_with_blank_length], NEG_INF),
        );
        let mut neg_log_likelihood = Tensor::<B, 1>::zeros([batch_size]);

        let mut target_iter = target_lengths
            .clone()
            .iter_dim(0)
            .scan(0usize, |start, current| {
                let step = current.into_scalar().elem::<u32>() as usize;
                let res = targets.clone().slice([*start..(*start + step)]);
                *start += step;

                Some(res)
            });

        for b in 0..batch_size {
            let target_data = target_iter.next().unwrap();

            let input_length = input_lengths
                .clone()
                .slice([b..(b + 1)])
                .into_scalar()
                .elem::<u32>() as usize;
            let [target_length] = target_data.dims();

            log_alphas = log_alphas.slice_assign(
                [b..(b + 1), 0..1, 0..1],
                log_probs
                    .clone()
                    .slice([b..(b + 1), 0..1, self.blank..(self.blank + 1)]),
            );

            if target_length > 0 {
                let target_prime = Self::get_target_prime(target_data.clone(), 1, self.blank);
                log_alphas = log_alphas.slice_assign(
                    [b..(b + 1), 0..1, 1..2],
                    log_probs
                        .clone()
                        .slice([b..(b + 1), 0..1, target_prime..(target_prime + 1)]),
                );
            }

            for t in 1..input_length {
                for s in 0..(2 * target_length + 1) {
                    let current_target_prime =
                        Self::get_target_prime(target_data.clone(), s, self.blank);

                    // \alpha_{t-1}(s)
                    let la1 = log_alphas
                        .clone()
                        .slice([b..(b + 1), (t - 1)..t, s..(s + 1)])
                        .reshape([1]);
                    // for the logsumexp calculation
                    let mut lamax = la1.clone();
                    // \alpha_{t-1}(s-1)
                    let (la2, la3);

                    if s > 0 {
                        la2 = log_alphas
                            .clone()
                            .slice([b..(b + 1), (t - 1)..t, (s - 1)..s])
                            .reshape([1]);
                        if la2.clone().greater(lamax.clone()).to_data().value[0] {
                            lamax = la2.clone();
                        }
                    } else {
                        la2 = Tensor::<B, 1>::full([1], NEG_INF);
                    }

                    if (s > 1)
                        && (Self::get_target_prime(target_data.clone(), s - 2, self.blank)
                            != current_target_prime)
                    {
                        // \alpha_{t-1}(s-2)
                        la3 = log_alphas
                            .clone()
                            .slice([b..(b + 1), (t - 1)..t, (s - 2)..(s - 1)])
                            .reshape([1]);
                        if la3.clone().greater(lamax.clone()).to_data().value[0] {
                            lamax = la3.clone();
                        }
                    } else {
                        la3 = Tensor::<B, 1>::full([1], NEG_INF);
                    }

                    if lamax.clone().equal_elem(NEG_INF).to_data().value[0] {
                        lamax = Tensor::<B, 1>::from_floats([0.0]);
                    }
                    log_alphas = log_alphas.slice_assign(
                        [b..(b + 1), t..(t + 1), s..(s + 1)],
                        (((la1 - lamax.clone()).exp()
                            + (la2 - lamax.clone()).exp()
                            + (la3 - lamax.clone()).exp())
                        .log()
                            + lamax
                            + log_probs
                                .clone()
                                .slice([
                                    b..(b + 1),
                                    t..(t + 1),
                                    current_target_prime..(current_target_prime + 1),
                                ])
                                .reshape([1]))
                        .reshape([1, 1, 1]),
                    );
                }
            }

            // the likelihood is the sum of the last two alphas,
            // the loss is the negative log likelihood
            if target_length == 0 {
                // if the target is empty then there is no preceding BLANK
                // state and hence there is no path to merge
                neg_log_likelihood = neg_log_likelihood.slice_assign(
                    [b..(b + 1)],
                    -log_alphas
                        .clone()
                        .slice([b..(b + 1), (input_length - 1)..input_length, 0..1])
                        .reshape([1]),
                );
            } else {
                let l1 = log_alphas
                    .clone()
                    .slice([
                        b..(b + 1),
                        (input_length - 1)..input_length,
                        (target_length * 2)..(target_length * 2 + 1),
                    ])
                    .reshape([1]);
                let l2 = log_alphas
                    .clone()
                    .slice([
                        b..(b + 1),
                        (input_length - 1)..input_length,
                        (target_length * 2 - 1)..(target_length * 2),
                    ])
                    .reshape([1]);
                // for the logsumexp calculation
                let mut m = Tensor::cat([l1.clone(), l2.clone()].to_vec(), 0).max();

                if m.clone().equal_elem(NEG_INF).to_data().value[0] {
                    m = Tensor::<B, 1>::from_floats([0.0])
                };
                let log_likelihood = ((l1 - m.clone()).exp() + (l2 - m.clone()).exp()).log() + m;
                neg_log_likelihood = neg_log_likelihood.slice_assign([b..(b + 1)], -log_likelihood);
            }
        }

        match reduction {
            Some(Reduction::Mean) | Some(Reduction::Auto) => {
                (neg_log_likelihood / target_lengths.float()).mean()
            }
            Some(Reduction::Sum) => neg_log_likelihood.sum(),
            None => neg_log_likelihood,
        }
    }

    fn get_target_prime(target_data: Tensor<B, 1, Int>, idx: usize, blank: usize) -> usize {
        if idx % 2 == 0 {
            blank
        } else {
            target_data
                .slice([(idx / 2)..(idx / 2 + 1)])
                .into_scalar()
                .elem::<u32>() as usize
        }
    }

    fn assertions(
        log_probs: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
        input_lengths: Tensor<B, 1, Int>,
        target_lengths: Tensor<B, 1, Int>,
    ) {
        let [log_probs_batch_size, input_seq_length, _] = log_probs.dims();
        let [targets_size] = targets.dims();
        let [input_lengths_size] = input_lengths.dims();
        let [target_lengths_size] = target_lengths.dims();

        assert!(
            log_probs_batch_size == input_lengths_size,
            "Batch size of log_probs ({}) should correspond to size of input_lengths ({}).",
            log_probs_batch_size,
            input_lengths
        );

        assert!(
            log_probs_batch_size == target_lengths_size,
            "Batch size of log_probs ({}) should correspond to size of target_lengths ({}).",
            log_probs_batch_size,
            target_lengths_size
        );

        assert!(
            target_lengths
                .sum()
                .equal_elem(targets_size as u32)
                .into_data()
                .value[0],
            "Batch size of targets ({}) should correspond to sum of target_lengths ({}).",
            log_probs_batch_size,
            target_lengths_size
        );

        let max_input_length = input_lengths.max();
        assert!(
            max_input_length.clone()
                .lower_equal_elem(input_seq_length as u32)
                .into_data()
                .value[0],
            "The maximum value of input_lengths ({}) must not be greater than the sequence length of log_probs ({}).",
            max_input_length.into_scalar(), input_seq_length
        );
    }
}

#[cfg(test)]
mod test {
    use burn_tensor::Data;

    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_ctc_loss() {
        let input = Tensor::<TestBackend, 3>::from_data([[
            [
                -3.941, -2.116, -3.559, -2.559, -2.576, -2.445, -0.759, -3.240, -3.116, -3.221,
            ],
            [
                -3.001, -2.181, -2.915, -2.382, -4.597, -4.133, -3.738, -3.256, -3.291, -0.571,
            ],
            [
                -1.127, -3.112, -2.896, -1.613, -4.025, -2.752, -2.086, -3.241, -2.187, -3.925,
            ],
            [
                -1.568, -4.852, -4.101, -3.584, -1.354, -2.619, -1.798, -3.845, -2.914, -1.789,
            ],
            [
                -3.770, -4.748, -3.915, -0.978, -6.070, -2.430, -3.295, -2.307, -3.980, -1.119,
            ],
            [
                -2.117, -2.178, -2.084, -2.325, -1.426, -3.922, -2.020, -4.461, -2.366, -3.078,
            ],
            [
                -2.195, -1.658, -2.019, -2.959, -3.266, -3.922, -1.259, -3.566, -2.426, -2.904,
            ],
            [
                -2.441, -1.606, -2.835, -3.703, -1.418, -3.456, -2.504, -2.445, -1.907, -3.263,
            ],
            [
                -3.509, -2.281, -2.405, -4.563, -2.469, -2.816, -1.916, -2.147, -1.701, -1.736,
            ],
            [
                -3.313, -1.417, -2.122, -3.138, -3.365, -2.074, -3.471, -1.530, -2.885, -2.362,
            ],
            [
                -3.784, -0.829, -2.479, -2.101, -3.563, -2.265, -4.733, -2.501, -2.731, -3.067,
            ],
            [
                -2.533, -2.684, -0.890, -2.986, -3.694, -3.484, -2.270, -2.169, -2.913, -2.751,
            ],
            [
                -3.435, -2.567, -2.526, -1.183, -3.210, -2.538, -1.184, -3.352, -3.935, -3.704,
            ],
            [
                -3.139, -2.204, -0.668, -5.249, -3.855, -3.706, -2.839, -1.971, -2.852, -3.608,
            ],
            [
                -1.445, -2.020, -3.576, -3.153, -2.949, -2.717, -3.902, -3.726, -1.594, -1.635,
            ],
            [
                -1.596, -4.902, -4.364, -4.571, -1.465, -3.689, -1.751, -2.032, -1.945, -2.764,
            ],
            [
                -3.326, -2.239, -2.965, -1.831, -2.958, -1.912, -1.695, -1.932, -2.353, -3.791,
            ],
            [
                -3.372, -2.850, -2.342, -0.841, -2.754, -3.297, -3.610, -2.152, -2.611, -2.760,
            ],
            [
                -2.843, -3.622, -1.551, -4.361, -4.325, -0.975, -3.459, -2.004, -2.758, -2.658,
            ],
            [
                -2.094, -3.114, -0.915, -3.207, -2.865, -2.215, -3.892, -4.120, -2.113, -2.693,
            ],
            [
                -3.049, -2.809, -3.370, -2.358, -2.038, -1.879, -1.957, -3.337, -2.198, -1.648,
            ],
            [
                -4.449, -2.300, -2.324, -3.414, -2.296, -1.620, -3.738, -2.128, -1.276, -3.311,
            ],
            [
                -2.133, -2.854, -2.711, -3.328, -3.735, -3.705, -0.627, -3.701, -4.156, -2.319,
            ],
            [
                -3.160, -3.321, -1.590, -3.735, -1.640, -3.614, -2.270, -1.911, -2.099, -2.314,
            ],
            [
                -3.044, -3.279, -1.939, -2.554, -2.272, -1.209, -2.627, -3.025, -2.187, -2.837,
            ],
            [
                -3.209, -3.186, -3.113, -2.002, -2.527, -2.561, -3.697, -2.347, -1.694, -1.282,
            ],
            [
                -1.297, -2.826, -2.052, -2.534, -2.544, -3.318, -2.015, -3.384, -2.755, -2.171,
            ],
            [
                -2.774, -2.740, -1.453, -3.754, -2.903, -2.309, -2.528, -1.664, -2.338, -2.345,
            ],
            [
                -3.036, -2.509, -0.726, -2.385, -4.339, -4.286, -3.388, -3.196, -3.755, -1.772,
            ],
            [
                -3.222, -3.674, -2.348, -2.324, -3.065, -2.748, -0.912, -2.595, -1.952, -4.408,
            ],
        ]]);
        let target = Tensor::<TestBackend, 1, Int>::from_data([3, 4, 7, 6, 3, 7, 3, 6, 2]);
        let input_lengths = Tensor::<TestBackend, 1, Int>::from_data([30]);
        let target_lengths = Tensor::<TestBackend, 1, Int>::from_data([9]);
        let expected_res = Data::from([47.73889923095703]);

        let ctc_loss = CTCLoss::<TestBackend>::new(0);
        let res = ctc_loss.forward(
            input,
            target,
            input_lengths,
            target_lengths,
            Some(Reduction::Sum),
        );

        // 47.7376
        res.to_data().assert_approx_eq(&expected_res, 2);
    }
}
