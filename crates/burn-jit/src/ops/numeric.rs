use crate::codegen::dialect::gpu::{BinaryOperator, Elem, Operator, Scope};
use crate::codegen::{
    Compilation, CompilationInfo, CompilationSettings, EagerHandle, Execution, InputInfo,
    OutputInfo, WorkgroupLaunch,
};
use crate::kernel::GpuComputeShaderPhase;
use crate::{binary, gpu, Compiler, Runtime};
use crate::{element::JitElement, tensor::JitTensor, unary};
use burn_compute::client::ComputeClient;
use burn_tensor::{ElementConversion, Shape};

pub fn full<R: Runtime, E: JitElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
    value: E,
) -> JitTensor<R, E, D> {
    let client = R::client(device);

    full_device::<R, E, D>(client, shape, device.clone(), value)
}

pub fn full_device<R: Runtime, E: JitElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    shape: Shape<D>,
    device: R::Device,
    value: E,
) -> JitTensor<R, E, D> {
    #[derive(new)]
    pub struct Ops<C, E> {
        _c: core::marker::PhantomData<C>,
        _e: core::marker::PhantomData<E>,
    }

    impl<C, E> GpuComputeShaderPhase for Ops<C, E>
    where
        C: Compiler,
        E: JitElement,
    {
        fn compile(&self) -> gpu::ComputeShader {
            let settings = CompilationSettings::default();
            let mut scope = gpu::Scope::root();
            let elem = E::gpu_elem();
            let op = gpu::Operator::Assign(gpu::UnaryOperator {
                input: scope.read_scalar(0, elem),
                out: scope.create_local(elem),
            });
            scope.register(op);

            let local = scope.last_local_index().unwrap().index().unwrap();
            let scalars = InputInfo::Scalar {
                elem: E::gpu_elem(),
                size: 1,
            };

            let out = OutputInfo::ArrayWrite {
                item: gpu::Item::Scalar(E::gpu_elem()),
                local,
            };
            let info = CompilationInfo {
                inputs: vec![scalars],
                outputs: vec![out],
                scope,
            };
            Compilation::new(info).compile(settings)
        }
    }

    let kernel = Ops::<R::Compiler, E>::new();
    let launch = WorkgroupLaunch::Input { pos: 0 };

    let num_elems = shape.num_elements();
    let buffer = client.empty(num_elems * core::mem::size_of::<E>());
    let output = JitTensor::new(client.clone(), device, shape, buffer);

    Execution::start(kernel, client.clone())
        .inputs(&[EagerHandle::<R>::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )])
        .with_scalars(&[value])
        .execute(launch);

    output
}

pub fn zeros<R: Runtime, E: JitElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
) -> JitTensor<R, E, D> {
    let client = R::client(device);

    zeros_device(client, device.clone(), shape)
}

pub fn zeros_device<R: Runtime, E: JitElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape<D>,
) -> JitTensor<R, E, D> {
    full_device::<R, E, D>(client, shape, device, 0.elem())
}

pub fn ones<R: Runtime, E: JitElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
) -> JitTensor<R, E, D> {
    let client = R::client(device);

    ones_device::<R, E, D>(client, device.clone(), shape)
}

pub fn ones_device<R: Runtime, E: JitElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape<D>,
) -> JitTensor<R, E, D> {
    full_device::<R, E, D>(client, shape, device, 1.elem())
}

pub fn empty_device<R: Runtime, E: JitElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape<D>,
) -> JitTensor<R, E, D> {
    let buffer = client.empty(shape.num_elements() * core::mem::size_of::<E>());

    JitTensor::new(client, device, shape, buffer)
}

pub fn add<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    binary!(
        operation: |scope: &mut Scope, elem: Elem| Operator::Add(BinaryOperator {
            lhs: scope.read_array(0, elem),
            rhs: scope.read_array(1, elem),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn add_scalar<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, E, D> {
    unary!(
        operation: |scope: &mut Scope, elem: Elem| Operator::Add(BinaryOperator {
            lhs: scope.read_array(0, elem),
            rhs: scope.read_scalar(0, elem),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn sub<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    binary!(
        operation: |scope: &mut Scope, elem: Elem| Operator::Sub(BinaryOperator {
            lhs: scope.read_array(0, elem),
            rhs: scope.read_array(1, elem),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn sub_scalar<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, E, D> {
    unary!(
        operation: |scope: &mut Scope, elem: Elem| Operator::Sub(BinaryOperator {
            lhs: scope.read_array(0, elem),
            rhs: scope.read_scalar(0, elem),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn mul<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    binary!(
        operation: |scope: &mut Scope, elem: Elem| Operator::Mul(BinaryOperator {
            lhs: scope.read_array(0, elem),
            rhs: scope.read_array(1, elem),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn mul_scalar<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, E, D> {
    unary!(
        operation: |scope: &mut Scope, elem: Elem| Operator::Mul(BinaryOperator {
            lhs: scope.read_array(0, elem),
            rhs: scope.read_scalar(0, elem),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn div<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    binary!(
        operation: |scope: &mut Scope, elem: Elem| Operator::Div(BinaryOperator {
            lhs: scope.read_array(0, elem),
            rhs: scope.read_array(1, elem),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn div_scalar<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, E, D> {
    unary!(
        operation: |scope: &mut Scope, elem: Elem| Operator::Div(BinaryOperator {
            lhs: scope.read_array(0, elem),
            rhs: scope.read_scalar(0, elem),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn remainder_scalar<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, E, D> {
    let shape = lhs.shape.clone();
    let device = lhs.device.clone();

    let rhs_tensor = full::<R, E, D>(shape, &device, rhs);

    binary!(
        operation: |scope: &mut Scope, elem: Elem| Operator::Remainder(BinaryOperator {
            lhs: scope.read_array(0, elem),
            rhs: scope.read_array(1, elem),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs_tensor,
        elem: E
    )
}

pub fn pow<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    binary!(
        operation: |scope: &mut Scope, elem: Elem| Operator::Powf(BinaryOperator {
            lhs: scope.read_array(0, elem),
            rhs: scope.read_array(1, elem),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}
