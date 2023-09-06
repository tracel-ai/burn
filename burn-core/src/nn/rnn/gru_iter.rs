use core::ops::Range;

use burn_tensor::activation::sigmoid;

use crate as burn;

use crate::nn::Initializer;
use crate::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

/// Confíguration for (Gru).
#[derive(Config)]
pub struct GruConfig {
    /// Size of the embedding for each time point.
    d_input: usize,

    /// Size for each time point in the output.
    d_hidden: usize,
    #[config(default = true)]
    bias: bool,

    /// Initializer for linear layers.
    #[config(default = "Initializer::XavierNormal{gain:1.0}")]
    initializer: Initializer,
}

impl GruConfig {
    /// Initialize (Gru) module.
    pub fn init<B: Backend>(&self) -> Gru<B> {
        let config = GateConfig::new(
            self.d_input,
            self.d_hidden,
            self.bias,
            self.initializer.clone(),
        );
        Gru {
            forget: config.init(),
            update_value: config.init(),
            update_factor: config.init(),
            d_hidden: self.d_hidden,
        }
    }

    /// Initialize (Gru) module from (GruRecord).
    pub fn init_with<B: Backend>(&self, record: GruRecord<B>) -> Gru<B> {
        let config = GateConfig::new(
            self.d_input,
            self.d_hidden,
            self.bias,
            self.initializer.clone(),
        );
        Gru {
            forget: config.init_with(record.forget),
            update_value: config.init_with(record.update_value),
            update_factor: config.init_with(record.update_factor),
            d_hidden: self.d_hidden,
        }
    }
}

/// Module implementing Gru architecture.
#[derive(Module, Debug)]
pub struct Gru<B: Backend> {
    forget: Gate<B>,
    update_value: Gate<B>,
    update_factor: Gate<B>,
    d_hidden: usize,
}

impl<B: Backend> Gru<B> {
    /// Returning a (GruIter) over the batched time series, whose output is the hidden state at each time point.
    pub fn over(&self, input: Tensor<B, 3>, state: Option<Tensor<B, 2>>) -> GruIter<B> {
        let [batch, seq, _] = input.dims();
        let ranges = input
            .dims()
            .into_iter()
            .map(|e| 0..e)
            .collect::<Vec<Range<usize>>>();
        let ranges: [Range<usize>; 3] = ranges.try_into().unwrap();
        GruIter {
            gru: self,
            counter: 0,
            ranges,
            end_idx: seq,
            input,
            state: state.unwrap_or(Tensor::<B, 2>::zeros([batch, self.d_hidden])),
        }
    }

    /// Applies the forward pass.
    pub fn forward(&self, input: Tensor<B, 2>, state: Tensor<B, 2>) -> Tensor<B, 2> {
        // Forget gate.
        let forget = sigmoid(self.forget.forward(input.clone(), state.clone()));

        // Update factor gate..
        let update_factor = sigmoid(self.update_factor.forward(input.clone(), state.clone()));

        // Output value gate.
        let update_value = self.update_value.forward(input.clone(), forget).tanh();

        state * (-update_factor.clone() + 1.) + update_value * update_factor
    }
}

/// Iterator for (Gru). Is returned by Gru::over;
#[derive(Debug)]
pub struct GruIter<'a, B: Backend> {
    gru: &'a Gru<B>,
    counter: usize,
    input: Tensor<B, 3>,
    state: Tensor<B, 2>,
    end_idx: usize,
    ranges: [Range<usize>; 3],
}

impl<'a, B: Backend> Iterator for GruIter<'a, B> {
    type Item = Tensor<B, 2>;

    fn next(&mut self) -> Option<Self::Item> {
        let res = if self.counter < self.end_idx {
            let mut ranges = self.ranges.clone();
            ranges[1] = self.counter..(self.counter + 1);
            let input = self.input.clone().slice(ranges).squeeze(1);
            self.state = self.gru.forward(input, self.state.clone());
            Some(self.state.clone())
        } else {
            None
        };
        self.counter += 1;
        res
    }
}

/// Config for (Gate).
#[derive(Config)]
pub struct GateConfig {
    d_input: usize,
    d_hidden: usize,
    bias: bool,
    initializer: Initializer,
}

impl GateConfig {
    /// Initialize (Gate) module.
    pub fn init<B: Backend>(&self) -> Gate<B> {
        Gate {
            input: LinearConfig::new(self.d_input, self.d_hidden)
                .with_bias(true)
                .with_initializer(self.initializer.clone())
                .init(),
            hidden: LinearConfig::new(self.d_hidden, self.d_hidden)
                .with_bias(true)
                .with_initializer(self.initializer.clone())
                .init(),
        }
    }

    /// Initialize (Gate) module from (GateRecord).
    pub fn init_with<B: Backend>(&self, record: GateRecord<B>) -> Gate<B> {
        Gate {
            input: LinearConfig::new(self.d_input, self.d_hidden).init_with(record.input),
            hidden: LinearConfig::new(self.d_hidden, self.d_hidden).init_with(record.hidden),
        }
    }
}

/// The Gate off a Gru.
#[derive(Module, Debug)]
pub struct Gate<B: Backend> {
    input: Linear<B>,
    hidden: Linear<B>,
}

impl<B: Backend> Gate<B> {
    /// Forward pass on Gate.
    pub fn forward(&self, input: Tensor<B, 2>, hidden: Tensor<B, 2>) -> Tensor<B, 2> {
        self.input.forward(input) + self.hidden.forward(hidden)
    }
}

#[cfg(test)]
mod test {
    use burn_tensor::Data;

    use crate::{
        module::{ConstantRecord, Param},
        nn::LinearRecord,
        TestBackend,
    };

    use super::*;

    /// Test forward pass with simple input vector.
    #[test]
    fn tests_forward_single_input_single_feature() {
        let record = GruRecord {
            forget: GateRecord {
                input: LinearRecord {
                    weight: Param::from(Tensor::<TestBackend, 2>::from_floats([[0.6]])),
                    bias: None,
                },
                hidden: LinearRecord {
                    weight: Param::from(Tensor::<TestBackend, 2>::from_floats([[0.]])),
                    bias: None,
                },
            },
            update_value: GateRecord {
                input: LinearRecord {
                    weight: Param::from(Tensor::<TestBackend, 2>::from_floats([[0.7]])),
                    bias: None,
                },
                hidden: LinearRecord {
                    weight: Param::from(Tensor::<TestBackend, 2>::from_floats([[0.]])),
                    bias: None,
                },
            },
            update_factor: GateRecord {
                input: LinearRecord {
                    weight: Param::from(Tensor::<TestBackend, 2>::from_floats([[0.6]])),
                    bias: None,
                },
                hidden: LinearRecord {
                    weight: Param::from(Tensor::<TestBackend, 2>::from_floats([[0.]])),
                    bias: None,
                },
            },
            d_hidden: ConstantRecord::new(),
        };

        let gru = GruConfig::new(1, 1).with_bias(false).init_with(record);
        let data = Tensor::<TestBackend, 3>::from_data(Data::from([[[0.1]]]));
        let output = gru.over(data, None).last().unwrap();

        output
            .to_data()
            .assert_approx_eq(&Data::from([[0.0359]]), 3); // Sigmoid(0.6 * 0.1) * Tanh(0.7 * 0.1) =~ 0.0359.
    }
}
