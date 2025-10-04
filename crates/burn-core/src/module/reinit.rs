use super::{Module, ModuleMapper};
use burn_tensor::{
    Element, ElementConversion, Tensor, TensorData,
    backend::Backend,
    ops::{FloatElem, IntElem},
};
use rand::{Rng, SeedableRng};

#[derive(Debug)]
/// Overrides float and int tensors of [burn modules](super::Module).
///
/// This is useful for testing.
pub struct Reinitializer<B: Backend> {
    float: ReinitStrategy<FloatElem<B>>,
    int: ReinitStrategy<IntElem<B>>,
}

#[derive(Debug)]
#[allow(missing_docs)]
enum ReinitStrategy<E> {
    Range { min: E, max: E },
    Constant { value: E },
    Random { seed: u64, min: E, max: E },
}

impl<B: Backend> Default for Reinitializer<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Reinitializer<B> {
    /// Create a new [reinitializer](Reinitializer).
    pub fn new() -> Self {
        Self {
            float: ReinitStrategy::Constant {
                value: 0.elem::<FloatElem<B>>(),
            },
            int: ReinitStrategy::Constant {
                value: 0.elem::<IntElem<B>>(),
            },
        }
    }

    /// Apply the reinitialization to the given [module](Module).
    pub fn apply<M: Module<B>>(mut self, module: M) -> M {
        module.map(&mut self)
    }

    /// Set the reinitialization strategy to constant for all tensors.
    pub fn constant(self, constant: f64) -> Self {
        self.constant_float(constant).constant_int(constant as i64)
    }

    /// Set the reinitialization strategy to constant for float tensors.
    pub fn constant_float(mut self, constant: f64) -> Self {
        self.float = ReinitStrategy::Constant {
            value: constant.elem(),
        };
        self
    }

    /// Set the reinitialization strategy to constant for int tensors.
    pub fn constant_int(mut self, constant: i64) -> Self {
        self.int = ReinitStrategy::Constant {
            value: constant.elem(),
        };
        self
    }
    /// Set the reinitialization strategy to random for all tensors.
    pub fn random(self, seed: u64, min: f64, max: f64) -> Self {
        self.random_float(seed, min, max)
            .random_int(seed, min as i64, max as i64)
    }

    /// Set the reinitialization strategy to random for float tensors.
    pub fn random_float(mut self, seed: u64, min: f64, max: f64) -> Self {
        self.float = ReinitStrategy::Random {
            seed,
            min: min.elem(),
            max: max.elem(),
        };
        self
    }

    /// Set the reinitialization strategy to random for int tensors.
    pub fn random_int(mut self, seed: u64, min: i64, max: i64) -> Self {
        self.int = ReinitStrategy::Random {
            seed,
            min: min.elem(),
            max: max.elem(),
        };
        self
    }

    /// Set the reinitialization strategy to range for all tensors.
    pub fn range(self, min: f64, max: f64) -> Self {
        self.range_float(min, max).range_int(min as i64, max as i64)
    }

    /// Set the reinitialization strategy to range for float tensors.
    pub fn range_float(mut self, min: f64, max: f64) -> Self {
        self.float = ReinitStrategy::Range {
            min: min.elem(),
            max: max.elem(),
        };
        self
    }

    /// Set the reinitialization strategy to range for int tensors.
    pub fn range_int(mut self, min: i64, max: i64) -> Self {
        self.int = ReinitStrategy::Range {
            min: min.elem(),
            max: max.elem(),
        };
        self
    }
}

impl<B: Backend> ModuleMapper<B> for Reinitializer<B> {
    fn map_float<const D: usize>(
        &mut self,
        param: super::Param<Tensor<B, D>>,
    ) -> super::Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let device = tensor.device();
        let shape = tensor.shape();
        let num_elements = shape.num_elements();

        let tensor = match &self.float {
            ReinitStrategy::Range { min, max } => {
                let tensor = Tensor::arange(0..num_elements as i64, &device)
                    .reshape(shape)
                    .float();
                let (factor, bias) = resolve::<FloatElem<B>>(*min, *max, num_elements);
                tensor * factor + bias
            }
            ReinitStrategy::Constant { value } => Tensor::full(shape, *value, &device),
            ReinitStrategy::Random { seed, min, max } => {
                let data = TensorData::new(
                    random_vector::<FloatElem<B>>(*seed, min.elem(), max.elem(), num_elements),
                    shape,
                );
                Tensor::from_data(data, &device)
            }
        };

        super::Param::into_initialized(id, tensor, mapper)
    }

    fn map_int<const D: usize>(
        &mut self,
        param: super::Param<Tensor<B, D, burn_tensor::Int>>,
    ) -> super::Param<Tensor<B, D, burn_tensor::Int>> {
        let (id, tensor, mapper) = param.consume();
        let device = tensor.device();
        let shape = tensor.shape();
        let num_elements = shape.num_elements();

        let tensor = match &self.int {
            ReinitStrategy::Range { min, max } => {
                let tensor = Tensor::arange(0..num_elements as i64, &device).reshape(shape);
                let (factor, bias) = resolve::<IntElem<B>>(*min, *max, num_elements);
                tensor * factor + bias
            }
            ReinitStrategy::Constant { value } => Tensor::full(shape, *value, &device),
            ReinitStrategy::Random { seed, min, max } => {
                let data = TensorData::new(
                    random_vector::<IntElem<B>>(*seed, min.elem(), max.elem(), num_elements),
                    shape,
                );
                Tensor::from_data(data, &device)
            }
        };

        super::Param::into_initialized(id, tensor, mapper)
    }

    fn map_bool<const D: usize>(
        &mut self,
        param: super::Param<Tensor<B, D, burn_tensor::Bool>>,
    ) -> super::Param<Tensor<B, D, burn_tensor::Bool>> {
        let (id, tensor, mapper) = param.consume();
        super::Param::into_initialized(id, tensor, mapper)
    }
}

fn resolve<E: Element>(min: E, max: E, num_elements: usize) -> (E, E) {
    let range = max.elem::<f64>() - min.elem::<f64>();
    let factor = range / num_elements as f64;
    let bias = min.elem::<f64>();

    (factor.elem(), bias.elem())
}

fn random_vector<E: Element>(seed: u64, min: f64, max: f64, num_elements: usize) -> Vec<E> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let dist = rand::distr::Uniform::new(min, max).unwrap();
    (0..num_elements)
        .map(|_| rng.sample(dist))
        .map(|e| e.elem::<E>())
        .collect()
}
