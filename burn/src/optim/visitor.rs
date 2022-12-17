use super::Optimizer;
use crate::module::{ModuleVisitor, ModuleVisitorMut, ParamId, StateNamed};
use burn_tensor::{backend::ADBackend, Tensor};

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
pub struct ModuleTensorUpdater<'a, B: ADBackend, O> {
    optimizer: &'a mut O,
    grads: &'a B::Gradients,
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
        self.optimizer.update_tensor(id, tensor, self.grads);
    }
}

impl<'a, B: ADBackend, O: Optimizer<Backend = B>> ModuleVisitor<B> for GradientsLoader<'a, B, O> {
    fn visit<const D: usize>(&mut self, id: &ParamId, tensor: &Tensor<B, D>) {
        self.optimizer
            .load_param_state::<D>(id, self.state, &tensor.device())
    }
}
