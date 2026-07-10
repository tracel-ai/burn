use burn_tensor::Tensor;

use crate::module::{ModuleMapper, Param, ParamGroup};

/// [ModuleMapper] that sets `require_grad` on the float parameters matching a [ParamGroup].
pub(crate) struct ParamGroupRequireGrad {
    path: Vec<String>,
    group: ParamGroup,
    require_grad: bool,
}

impl ParamGroupRequireGrad {
    pub(crate) fn new(group: ParamGroup, require_grad: bool) -> Self {
        Self {
            path: Vec::new(),
            group,
            require_grad,
        }
    }
}

impl ModuleMapper for ParamGroupRequireGrad {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (id, mut tensor, mapper) = param.consume();
        let path = self.path.join(".");
        if self.group.matches(&id, Some(&path)) {
            tensor = tensor.set_require_grad(self.require_grad);
        }
        Param::from_mapped_value(id, tensor, mapper)
    }
}
