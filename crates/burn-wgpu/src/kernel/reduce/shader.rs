use std::marker::PhantomData;

use crate::{
    codegen::{
        dialect::gpu::{gpu, Elem, Item, Scope, Variable, Visibility},
        Compilation, CompilationInfo, CompilationSettings, Compiler, InputInfo, OutputInfo,
    },
    element::JitElement,
    kernel::{DynamicKernelSource, SourceTemplate},
    Runtime,
};

pub(crate) trait ReduceDim: Send + Sync + 'static {
    fn initialize(scope: &mut Scope, output_item: Item) -> Variable;
    fn inner_loop(scope: &mut Scope, accumulator: Variable, current_value: Variable);
    fn assign(
        scope: &mut Scope,
        output: Variable,
        accumulator: Variable,
        shape_reduce_dim: Variable,
    );
}

pub(crate) struct SumDim;

impl ReduceDim for SumDim {
    fn initialize(scope: &mut Scope, output_item: Item) -> Variable {
        scope.zero(output_item)
    }

    fn inner_loop(scope: &mut Scope, accumulator: Variable, value: Variable) {
        gpu!(scope, accumulator += value);
    }

    fn assign(
        scope: &mut Scope,
        output: Variable,
        accumulator: Variable,
        _shape_reduce_dim: Variable,
    ) {
        let id = Variable::Id;
        gpu!(scope, output[id] = accumulator);
    }
}

pub(crate) struct MeanDim;

impl ReduceDim for MeanDim {
    fn initialize(scope: &mut Scope, output_item: Item) -> Variable {
        scope.zero(output_item)
    }

    fn inner_loop(scope: &mut Scope, accumulator: Variable, value: Variable) {
        gpu!(scope, accumulator += value);
    }

    fn assign(
        scope: &mut Scope,
        output: Variable,
        accumulator: Variable,
        shape_reduce_dim: Variable,
    ) {
        let id = Variable::Id;
        let denominator = scope.create_local(accumulator.item());
        gpu!(scope, denominator = cast(shape_reduce_dim));
        gpu!(scope, accumulator = accumulator / denominator);
        gpu!(scope, output[id] = accumulator);
    }
}

pub(crate) struct ReduceDimComputeShader<RD: ReduceDim> {
    tensor: Variable,
    dim: usize,
    output: Variable,
    reduce_dim: PhantomData<RD>,
}

#[derive(new)]
pub(crate) struct ReduceDimEagerKernel<RD: ReduceDim, R: Runtime, E: JitElement> {
    dim: usize,
    reduce_dim: PhantomData<RD>,
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

impl<RD: ReduceDim, R: Runtime, E: JitElement> DynamicKernelSource
    for ReduceDimEagerKernel<RD, R, E>
{
    fn source(&self) -> crate::kernel::SourceTemplate {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();

        let tensor = Variable::GlobalInputArray(0, item);

        let output = Variable::GlobalOutputArray(0, item);

        ReduceDimComputeShader {
            tensor,
            dim: self.dim,
            output,
            reduce_dim: PhantomData::<RD>,
        }
        .expand(&mut scope);

        scope.write_global_custom(output);

        let tensor = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };

        let out = OutputInfo::Array { item };

        let info = CompilationInfo {
            inputs: vec![tensor],
            outputs: vec![out],
            scope,
        };

        let settings = CompilationSettings::default();
        let shader = Compilation::new(info).compile(settings);
        let shader = <R::Compiler as Compiler>::compile(shader);
        SourceTemplate::new(shader.to_string())
    }

    fn id(&self) -> String {
        format!("{:?}dim={}", core::any::TypeId::of::<Self>(), self.dim)
    }
}

impl<RD: ReduceDim> ReduceDimComputeShader<RD> {
    pub(crate) fn expand(self, scope: &mut Scope) {
        let tensor = self.tensor;
        let dim: Variable = self.dim.into();
        let id = Variable::Id;
        let output = self.output;

        let offset_input = scope.zero(Elem::UInt);
        let stride_input_dim = scope.create_local(Elem::UInt);
        let shape_input_dim = scope.create_local(Elem::UInt);

        gpu!(
            scope,
            range(0u32, Variable::Rank).for_each(|i, scope| {
                let stride_input = scope.create_local(Elem::UInt);
                let stride_output = scope.create_local(Elem::UInt);
                let shape_output = scope.create_local(Elem::UInt);

                gpu!(scope, stride_input = stride(tensor, i));
                gpu!(scope, stride_output = stride(output, i));
                gpu!(scope, shape_output = shape(output, i));

                let offset_local = scope.create_local(Elem::UInt);
                gpu!(scope, offset_local = id / stride_output);
                gpu!(scope, offset_local = offset_local % shape_output);

                let is_dim_reduce = scope.create_local(Elem::Bool);
                gpu!(scope, is_dim_reduce = i == dim);

                gpu!(scope, if(is_dim_reduce).then(|scope|{
                    gpu!(scope, shape_input_dim = shape(tensor, i));
                    gpu!(scope, stride_input_dim = stride_input);
                    gpu!(scope, offset_input += offset_local);
                }).else(|scope|{
                    gpu!(scope, offset_local = offset_local * stride_input);
                    gpu!(scope, offset_input += offset_local);
                }));
            })
        );

        let accumulator = RD::initialize(scope, output.item());

        gpu!(
            scope,
            range(0u32, shape_input_dim).for_each(|i, scope| {
                let index = scope.create_local(Elem::UInt);
                gpu!(scope, index = i * stride_input_dim);
                gpu!(scope, index += offset_input);
                let value = scope.create_local(tensor.item());
                gpu!(scope, value = tensor[index]);
                RD::inner_loop(scope, accumulator, value);
            })
        );

        RD::assign(scope, output, accumulator, shape_input_dim);
    }
}
