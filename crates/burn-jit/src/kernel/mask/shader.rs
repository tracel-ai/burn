use burn_cube::{
    cpa,
    ir::{Elem, IndexOffsetGlobalWithLayout, Item, KernelDefinition, Scope, Variable, Visibility},
    InputInfo, KernelExpansion, KernelIntegrator, KernelSettings, OutputInfo,
};
use std::marker::PhantomData;

use crate::{kernel::Kernel, JitElement, JitRuntime};

pub(crate) trait MaskStrategy: Send + Sync + 'static {
    fn mask(
        scope: &mut Scope,
        masked_value: Variable,
        value: Variable,
        index: Variable,
    ) -> Variable;

    fn value_info(value_item: Item) -> InputInfo;
    fn value_variable(value_item: Item) -> Variable;
}

pub(crate) struct MaskFill;

impl MaskStrategy for MaskFill {
    fn mask(
        scope: &mut Scope,
        masked_value: Variable,
        value: Variable,
        _index: Variable,
    ) -> Variable {
        cpa!(scope, masked_value = value);
        masked_value
    }

    fn value_info(value_item: Item) -> InputInfo {
        InputInfo::Scalar {
            elem: value_item.elem(),
            size: 1,
        }
    }

    fn value_variable(value_item: Item) -> Variable {
        Variable::GlobalScalar(0, value_item.elem())
    }
}

pub(crate) struct MaskWhere;

impl MaskStrategy for MaskWhere {
    fn mask(
        scope: &mut Scope,
        masked_value: Variable,
        value: Variable,
        index: Variable,
    ) -> Variable {
        cpa!(scope, masked_value = value[index]);
        masked_value
    }

    fn value_info(value_item: Item) -> InputInfo {
        InputInfo::Array {
            item: value_item,
            visibility: Visibility::Read,
        }
    }

    fn value_variable(value_item: Item) -> Variable {
        Variable::GlobalInputArray(2, value_item)
    }
}

pub(crate) struct MaskShader<EI: JitElement, EM: JitElement, M: MaskStrategy> {
    input: Variable,
    mask: Variable,
    value: Variable,
    output: Variable,
    reversed: bool,
    _mask_strategy: PhantomData<M>,
    _input_elem: PhantomData<EI>,
    _mask_elem: PhantomData<EM>,
}

#[derive(new)]
pub(crate) struct MaskReadOnlyEagerKernel<
    M: MaskStrategy,
    R: JitRuntime,
    EI: JitElement,
    EM: JitElement,
> {
    reversed: bool,
    _mask_strategy: PhantomData<M>,
    _runtime: PhantomData<R>,
    _input_elem: PhantomData<EI>,
    _mask_elem: PhantomData<EM>,
}

impl<M: MaskStrategy, R: JitRuntime, EI: JitElement, EM: JitElement> Kernel
    for MaskReadOnlyEagerKernel<M, R, EI, EM>
{
    fn define(&self) -> KernelDefinition {
        let mut scope = Scope::root();
        let tensor_item = EI::cube_elem().into();
        let mask_item = EM::cube_elem().into();

        let input = Variable::GlobalInputArray(0, tensor_item);
        let mask = Variable::GlobalInputArray(1, mask_item);
        let value = M::value_variable(tensor_item);
        let output = Variable::GlobalOutputArray(0, tensor_item);

        MaskShader::<EI, EM, M> {
            input,
            mask,
            value,
            output,
            reversed: self.reversed,
            _mask_strategy: PhantomData::<M>,
            _input_elem: PhantomData::<EI>,
            _mask_elem: PhantomData::<EM>,
        }
        .expand(&mut scope);

        scope.write_global_custom(output);

        let input = InputInfo::Array {
            item: tensor_item,
            visibility: Visibility::Read,
        };

        let mask = InputInfo::Array {
            item: mask_item,
            visibility: Visibility::Read,
        };

        let value = M::value_info(tensor_item);

        let out = OutputInfo::Array { item: tensor_item };

        let info = KernelExpansion {
            inputs: vec![input, mask, value],
            outputs: vec![out],
            scope,
        };

        let settings = KernelSettings::default();
        KernelIntegrator::new(info).integrate(settings)
    }

    fn id(&self) -> String {
        format!(
            "{:?} rev={}",
            core::any::TypeId::of::<Self>(),
            self.reversed
        )
    }
}

#[derive(new)]
pub(crate) struct MaskInplaceEagerKernel<
    M: MaskStrategy,
    R: JitRuntime,
    EI: JitElement,
    EM: JitElement,
> {
    reversed: bool,
    _mask_strategy: PhantomData<M>,
    _runtime: PhantomData<R>,
    _input_elem: PhantomData<EI>,
    _mask_elem: PhantomData<EM>,
}

impl<M: MaskStrategy, R: JitRuntime, EI: JitElement, EM: JitElement> Kernel
    for MaskInplaceEagerKernel<M, R, EI, EM>
{
    fn define(&self) -> KernelDefinition {
        let mut scope = Scope::root();
        let tensor_item = EI::cube_elem().into();
        let mask_item = EM::cube_elem().into();

        let input = Variable::GlobalInputArray(0, tensor_item);
        let mask = Variable::GlobalInputArray(1, mask_item);
        let value = M::value_variable(tensor_item);

        MaskShader::<EI, EM, M> {
            input,
            mask,
            value,
            output: input,
            reversed: self.reversed,
            _mask_strategy: PhantomData::<M>,
            _input_elem: PhantomData::<EI>,
            _mask_elem: PhantomData::<EM>,
        }
        .expand(&mut scope);

        let input = InputInfo::Array {
            item: tensor_item,
            visibility: Visibility::ReadWrite,
        };

        let mask = InputInfo::Array {
            item: mask_item,
            visibility: Visibility::Read,
        };

        let value = M::value_info(tensor_item);

        let info = KernelExpansion {
            inputs: vec![input, mask, value],
            outputs: vec![],
            scope,
        };

        let settings = KernelSettings::default();
        KernelIntegrator::new(info).integrate(settings)
    }

    fn id(&self) -> String {
        format!(
            "{:?} rev={}",
            core::any::TypeId::of::<Self>(),
            self.reversed
        )
    }
}

impl<EI: JitElement, EM: JitElement, M: MaskStrategy> MaskShader<EI, EM, M> {
    pub(crate) fn expand(self, scope: &mut Scope) {
        let id = Variable::AbsolutePos;
        let input = self.input;
        let mask = self.mask;
        let value = self.value;
        let output = self.output;

        let index_input = scope.zero(Elem::UInt);
        let index_mask = scope.zero(Elem::UInt);

        IndexOffsetGlobalWithLayout {
            tensors: vec![input, mask],
            indexes: vec![index_input, index_mask],
            layout: output,
            position: id,
            dim_start: 0u32.into(),
            dim_end: Variable::Rank,
        }
        .expand(scope);

        // Determine if index should be masked
        let value_in_mask = scope.create_local(mask.item());
        cpa!(scope, value_in_mask = mask[index_mask]);
        let masked = scope.create_local(Elem::Bool);
        let zero = scope.zero(value_in_mask.item());
        if self.reversed {
            cpa!(scope, masked = value_in_mask == zero);
        } else {
            cpa!(scope, masked = value_in_mask != zero);
        }

        // Assign a value at the index
        let used_value = scope.create_local(output.item());
        cpa!(scope, if(masked).then(|scope| {
            M::mask(scope, used_value, value, index_input );
        }).else(|scope| {
            cpa!(scope, used_value = input[index_input]);
        }));
        cpa!(scope, output[id] = used_value);
    }
}
