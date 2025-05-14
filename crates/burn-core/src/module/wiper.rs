use burn_tensor::{
    Element, ElementConversion, Tensor,
    backend::Backend,
    ops::{FloatElem, IntElem},
};

use super::{ModuleMapper, ParamId};

#[derive(new, Debug)]
pub struct ModuleWiper<B: Backend> {
    floats: WipeStrategy<FloatElem<B>>,
    ints: WipeStrategy<IntElem<B>>,
}

#[derive(Debug)]
pub enum WipeStrategy<E> {
    Arange(Option<WipeNormalization<E>>),
    Constant(E),
}

#[derive(Debug)]
pub enum WipeNormalization<E> {
    Fixed { bias: E, factor: E },
    Uniform { min: E, max: E },
}

impl<E: Element> WipeNormalization<E> {
    fn resolve(&self, num_elements: usize) -> (E, E) {
        match self {
            WipeNormalization::Fixed { bias, factor } => (factor.clone(), bias.clone()),
            WipeNormalization::Uniform { min, max } => {
                let range = max.elem::<f64>() - min.elem::<f64>();
                let factor = range / num_elements as f64;
                let bias = min.elem::<f64>();
                println!("factor={factor} bias={bias} num_elements={num_elements}");

                (factor.elem(), bias.elem())
            }
        }
    }
}

impl<B: Backend> ModuleMapper<B> for ModuleWiper<B> {
    fn map_float<const D: usize>(&mut self, _id: ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let device = tensor.device();
        let shape = tensor.shape();
        let num_elements = shape.num_elements();

        match &self.floats {
            WipeStrategy::Arange(norm) => {
                let tensor = Tensor::arange(0..num_elements as i64, &device)
                    .reshape(shape)
                    .float();
                let norm = match norm {
                    None => return tensor,
                    Some(val) => val,
                };
                let (factor, bias) = norm.resolve(num_elements);
                tensor * factor + bias
            }
            WipeStrategy::Constant(value) => Tensor::full(shape, value.clone(), &device),
        }
    }

    fn map_int<const D: usize>(
        &mut self,
        _id: ParamId,
        tensor: Tensor<B, D, burn_tensor::Int>,
    ) -> Tensor<B, D, burn_tensor::Int> {
        let device = tensor.device();
        let shape = tensor.shape();
        let num_elements = shape.num_elements();

        match &self.ints {
            WipeStrategy::Arange(norm) => {
                let tensor = Tensor::arange(0..num_elements as i64, &device).reshape(shape);
                let norm = match norm {
                    None => return tensor,
                    Some(val) => val,
                };
                let (factor, bias) = norm.resolve(num_elements);
                tensor * factor + bias
            }
            WipeStrategy::Constant(value) => Tensor::full(shape, value.clone(), &device),
        }
    }

    fn map_bool<const D: usize>(
        &mut self,
        _id: ParamId,
        tensor: Tensor<B, D, burn_tensor::Bool>,
    ) -> Tensor<B, D, burn_tensor::Bool> {
        tensor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_uniform() {
        let norm = WipeNormalization::Uniform { min: -5.0, max: 5.0 };
        let (factor, bias) = norm.resolve(64);

        assert_eq!(bias, -5.0);
        assert_eq!(factor, 10.0 / 64.0);
    }
}
