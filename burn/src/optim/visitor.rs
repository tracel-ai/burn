use super::Optimizer;
use crate::module::{ADModule, ModuleVisitor, ModuleVisitorMut, ParamId, StateNamed};
use burn_tensor::{
    backend::{ADBackend, Backend},
    container::TensorContainer,
    Tensor,
};

pub type GradientsParams<B> = TensorContainer<<B as ADBackend>::InnerBackend, ParamId>;

#[derive(new)]
pub struct GradientsRegister<'a, B: ADBackend, O> {
    optimizer: &'a O,
    state: &'a mut StateNamed<B::Elem>,
}

#[derive(new)]
pub struct GradientsLoader<'a, B: ADBackend, O> {
    optimizer: &'a mut O,
    state: &'a StateNamed<B::Elem>,
}

#[derive(new)]
pub struct GradientsParamsConverter<'a, B: ADBackend> {
    grads: B::Gradients,
    grads_params: &'a mut TensorContainer<B::InnerBackend, ParamId>,
}

#[derive(new)]
pub struct ModuleTensorUpdater<'a, B: ADBackend, O> {
    optimizer: &'a mut O,
    grads: GradientsParams<B>,
}

#[derive(new)]
pub struct GradientsParamsChangeDevice<'a, B: ADBackend> {
    device: B::Device,
    grads: &'a mut GradientsParams<B>,
}

#[derive(new)]
pub struct ParamIdCollector<'a> {
    grads: &'a mut Vec<ParamId>,
}

impl<'a, B: ADBackend, O: Optimizer<Backend = B>> ModuleVisitor<B> for GradientsRegister<'a, B, O> {
    fn visit<const D: usize>(&mut self, id: &ParamId, _tensor: &Tensor<B, D>) {
        self.optimizer.register_param_state::<D>(id, self.state)
    }
}

impl<'a, B: ADBackend, O: Optimizer<Backend = B>> ModuleVisitorMut<B>
    for ModuleTensorUpdater<'a, B, O>
{
    fn visit_mut<const D: usize>(&mut self, id: &ParamId, tensor: &mut Tensor<B, D>) {
        if let Some(grad) = self.grads.get(id) {
            self.optimizer.update_tensor(id, tensor, grad);
        }
    }
}

impl<'a, B: ADBackend, O: Optimizer<Backend = B>> ModuleVisitor<B> for GradientsLoader<'a, B, O> {
    fn visit<const D: usize>(&mut self, id: &ParamId, tensor: &Tensor<B, D>) {
        self.optimizer
            .load_param_state::<D>(id, self.state, &tensor.device())
    }
}

impl<'a, B: ADBackend> ModuleVisitor<B> for ParamIdCollector<'a> {
    fn visit<const D: usize>(&mut self, id: &ParamId, _tensor: &Tensor<B, D>) {
        self.grads.push(id.clone());
    }
}

impl<'a, B: ADBackend> ModuleVisitor<B> for GradientsParamsConverter<'a, B> {
    fn visit<const D: usize>(&mut self, id: &ParamId, tensor: &Tensor<B, D>) {
        if let Some(grad) = tensor.grad(&self.grads) {
            self.grads_params.register(id.clone(), grad);
        }
    }
}

impl<'a, B: ADBackend> ModuleVisitor<B> for GradientsParamsChangeDevice<'a, B> {
    fn visit<const D: usize>(&mut self, id: &ParamId, _tensor: &Tensor<B, D>) {
        if let Some(grad) = self.grads.remove::<D>(id) {
            self.grads.register(id.clone(), grad.to_device(self.device));
        }
    }
}

pub fn to_device_grads<M: ADModule>(
    grads: &mut GradientsParams<M::ADBackend>,
    device: <M::Backend as Backend>::Device,
    module: &M,
) {
    let mut visitor = GradientsParamsChangeDevice::new(device, grads);
    module.visit(&mut visitor);
}

/// Convert the gradients returned by the ADBackend into a tensor container that contains
/// gradients corresponding to the given module.
pub fn convert_grads<M: ADModule>(
    grads: <M::ADBackend as ADBackend>::Gradients,
    module: &M,
) -> GradientsParams<M::ADBackend> {
    let mut grads_params = TensorContainer::new();
    let mut visitor = GradientsParamsConverter::new(grads, &mut grads_params);
    module.visit(&mut visitor);

    grads_params
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        module::Module,
        nn::{Linear, LinearConfig},
        TestADBackend,
    };
    use burn_tensor::{backend::Backend, Distribution};

    #[test]
    fn test_convert_grads_to_params_id() {
        let layer_1 = layer();
        let mut layer_2 = layer_1.clone();
        layer_2.to_device(<TestADBackend as Backend>::Device::default());
        layer_2.detach();
        let loss_1 = layer_1.forward(random_tensor());
        let loss_2 = layer_2.forward(random_tensor());
        let mut params_ids_1 = Vec::new();
        let mut params_ids_2 = Vec::new();
        let mut visitor_1 = ParamIdCollector::new(&mut params_ids_1);
        let mut visitor_2 = ParamIdCollector::new(&mut params_ids_2);
        let grads_1 = loss_1.backward();
        let grads_2 = loss_2.backward();

        layer_1.visit(&mut visitor_1);
        layer_2.visit(&mut visitor_2);

        convert_grads(grads_1, &layer_1);
        convert_grads(grads_2, &layer_2);

        layer_1.visit(&mut visitor_1);
        layer_2.visit(&mut visitor_2);

        assert_eq!(params_ids_1, params_ids_2);
    }

    fn layer() -> Linear<TestADBackend> {
        Linear::<TestADBackend>::new(&LinearConfig {
            d_input: 20,
            d_output: 20,
            bias: true,
        })
    }

    fn random_tensor() -> Tensor<TestADBackend, 2> {
        Tensor::<TestADBackend, 2>::random([2, 20], Distribution::Standard)
    }
}
