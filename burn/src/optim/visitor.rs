use super::Optimizer;
use crate::module::{ADModule, ModuleVisitor, ModuleVisitorMut, ParamId, StateNamed};
use burn_tensor::{
    backend::{ADBackend, Gradients},
    Tensor,
};

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
    grads: &'a mut B::Gradients,
}

#[derive(new)]
pub struct ModuleTensorUpdater<'a, B: ADBackend, O> {
    optimizer: &'a mut O,
    grads: &'a B::Gradients,
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
        if let Some(grad) = param_grad(id, tensor, self.grads) {
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
        if let Some(grad) = self.grads.remove::<D>(&tensor.node_id()) {
            self.grads.register(id.to_string(), grad);
        }
    }
}

pub fn param_grad<B: ADBackend, const D: usize>(
    id: &ParamId,
    tensor: &Tensor<B, D>,
    grads: &B::Gradients,
) -> Option<Tensor<B::InnerBackend, D>> {
    if let Some(grad) = grads.get(&id.to_string()) {
        return Some(grad);
    }

    if let Some(grad) = tensor.grad(grads) {
        return Some(grad);
    }

    None
}

pub fn convert_grads_to_param<M: ADModule>(
    grads: &mut <M::ADBackend as ADBackend>::Gradients,
    module: &M,
) {
    let mut visitor = GradientsParamsConverter::new(grads);
    module.visit(&mut visitor);
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
        let mut grads_1 = loss_1.backward();
        let mut grads_2 = loss_2.backward();

        layer_1.visit(&mut visitor_1);
        layer_2.visit(&mut visitor_2);

        convert_grads_to_param(&mut grads_1, &layer_1);
        convert_grads_to_param(&mut grads_2, &layer_2);

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
