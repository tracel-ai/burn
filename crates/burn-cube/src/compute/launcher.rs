use crate::compute::{CubeCount, KernelTask};
use crate::ir::{Elem, FloatKind, IntKind};
use crate::prelude::ArrayHandle;
use crate::{calculate_num_elems_dyn_rank, frontend::TensorHandle, Kernel, Runtime};
use burn_compute::client::ComputeClient;
use burn_compute::server::Binding;
use bytemuck::NoUninit;
use num_traits::ToPrimitive;

/// Prepare a kernel for [launch](KernelLauncher::launch).
pub struct KernelLauncher<R: Runtime> {
    tensors: TensorState<R>,
    scalar_bf16: ScalarState<half::bf16>,
    scalar_f16: ScalarState<half::f16>,
    scalar_f32: ScalarState<f32>,
    scalar_f64: ScalarState<f64>,
    scalar_u32: ScalarState<u32>,
    scalar_i64: ScalarState<i64>,
    scalar_i32: ScalarState<i32>,
    scalar_order: Vec<Elem>,
}

impl<R: Runtime> KernelLauncher<R> {
    /// Register a tensor to be launched.
    pub fn register_tensor(&mut self, tensor: &TensorHandle<'_, R>) {
        self.tensors.push(tensor);
    }

    /// Register an array to be launched.
    pub fn register_array(&mut self, array: &ArrayHandle<'_, R>) {
        self.tensors.push(&array.as_tensor());
    }

    /// Register a u32 scalar to be launched.
    pub fn register_u32(&mut self, scalar: u32) {
        self.register_scalar(Elem::UInt);
        self.scalar_u32.push(scalar);
    }

    /// Register a i32 scalar to be launched.
    pub fn register_i32(&mut self, scalar: i32) {
        self.register_scalar(Elem::Int(IntKind::I32));
        self.scalar_i32.push(scalar);
    }

    /// Register a i64 scalar to be launched.
    pub fn register_i64(&mut self, scalar: i64) {
        self.register_scalar(Elem::Int(IntKind::I64));
        self.scalar_i64.push(scalar);
    }

    /// Register a bf16 scalar to be launched.
    pub fn register_bf16(&mut self, scalar: half::bf16) {
        self.register_scalar(Elem::Float(FloatKind::BF16));
        self.scalar_bf16.push(scalar);
    }

    /// Register a f16 scalar to be launched.
    pub fn register_f16(&mut self, scalar: half::f16) {
        self.register_scalar(Elem::Float(FloatKind::F16));
        self.scalar_f16.push(scalar);
    }

    /// Register a f32 scalar to be launched.
    pub fn register_f32(&mut self, scalar: f32) {
        self.register_scalar(Elem::Float(FloatKind::F32));
        self.scalar_f32.push(scalar);
    }

    /// Register a f64 scalar to be launched.
    pub fn register_f64(&mut self, scalar: f64) {
        self.register_scalar(Elem::Float(FloatKind::F64));
        self.scalar_f64.push(scalar);
    }

    /// Launch the kernel.
    pub fn launch<K: Kernel>(
        self,
        cube_count: CubeCount,
        kernel: K,
        client: ComputeClient<R::Server, R::Channel>,
    ) {
        let bindings = self.into_bindings(&client);

        let kernel = Box::new(KernelTask::<R::Compiler, K>::new(kernel, cube_count));

        client.execute(kernel, bindings);
    }

    /// We need to create the bindings in the same order they are defined in the compilation step.
    ///
    /// The function [crate::KernelIntegrator::integrate] stars by registering the input tensors followed
    /// by the output tensors. Then the tensor metadata, and the scalars at the end. The scalars
    /// are registered in the same order they are added. This is why we store the scalar data type
    /// in the `scalar_order` vector, so that we can register them in the same order.
    fn into_bindings(
        mut self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Vec<Binding<R::Server>> {
        let mut bindings = Vec::new();

        self.tensors.register(client, &mut bindings);

        for elem in self.scalar_order.drain(..) {
            match elem {
                Elem::Float(kind) => match kind {
                    FloatKind::F16 => self.scalar_f16.register::<R>(client, &mut bindings),
                    FloatKind::BF16 => self.scalar_bf16.register::<R>(client, &mut bindings),
                    FloatKind::F32 => self.scalar_f32.register::<R>(client, &mut bindings),
                    FloatKind::F64 => self.scalar_f64.register::<R>(client, &mut bindings),
                },
                Elem::Int(kind) => match kind {
                    IntKind::I32 => self.scalar_i32.register::<R>(client, &mut bindings),
                    IntKind::I64 => self.scalar_i64.register::<R>(client, &mut bindings),
                },
                Elem::UInt => self.scalar_u32.register::<R>(client, &mut bindings),
                Elem::Bool => panic!("Bool can't be passed as bindings."),
            }
        }

        bindings
    }

    fn register_scalar(&mut self, elem: Elem) {
        if !self.scalar_order.contains(&elem) {
            self.scalar_order.push(elem);
        }
    }
}

/// Handles the tensor state.
pub enum TensorState<R: Runtime> {
    /// No tensor is registered yet.
    Empty,
    /// The registered tensors.
    Some {
        bindings: Vec<Binding<R::Server>>,
        metadata: Vec<u32>,
        lengths: Vec<u32>,
    },
}

/// Handles the scalar state of an element type
///
/// The scalars are grouped to reduce the number of buffers needed to send data to the compute device.
pub enum ScalarState<T> {
    /// No scalar of that type is registered yet.
    Empty,
    /// The registered scalars.
    Some(Vec<T>),
}

impl<R: Runtime> TensorState<R> {
    /// Push a new tensor to the state.
    pub fn push(&mut self, tensor: &TensorHandle<'_, R>) {
        if let TensorState::Empty = self {
            *self = TensorState::Some {
                bindings: Vec::with_capacity(1),
                metadata: Vec::new(),
                lengths: Vec::new(),
            };
        };

        let (bindings, metadata, lengths) = match self {
            TensorState::Empty => panic!("Should be init"),
            TensorState::Some {
                bindings,
                metadata,
                lengths,
            } => (bindings, metadata, lengths),
        };

        bindings.push(tensor.handle.clone().binding());

        let old_rank = if metadata.is_empty() {
            let rank = tensor.strides.len() as u32;
            metadata.push(rank);
            None
        } else if tensor.strides.len() > metadata[0] as usize {
            let old_rank = metadata[0];
            let rank = tensor.strides.len() as u32;
            Self::adjust_rank(metadata, bindings.len(), rank);
            Some(old_rank)
        } else {
            None
        };

        Self::register_strides(tensor.strides, tensor.shape, old_rank, metadata);
        Self::register_shape(tensor.shape, old_rank, metadata);

        if R::require_array_lengths() {
            let len = calculate_num_elems_dyn_rank(tensor.shape);
            lengths.push(len as u32);
        }
    }

    fn adjust_rank(metadata: &mut Vec<u32>, num_registered: usize, rank: u32) {
        let old_rank = metadata[0] as usize;
        let rank_diff = rank as usize - old_rank;
        let mut updated_metadata = Vec::with_capacity(2 * rank_diff * num_registered);

        for pos in 0..num_registered {
            let stride_index = (pos * old_rank * 2) + 1;
            let shape_index = stride_index + old_rank;

            let strides_old = &metadata[stride_index..stride_index + old_rank];
            let shape_old = &metadata[shape_index..shape_index + old_rank];

            Self::register_strides(
                strides_old,
                shape_old,
                Some(old_rank as u32),
                &mut updated_metadata,
            );
            Self::register_shape(shape_old, Some(old_rank as u32), &mut updated_metadata);
        }

        core::mem::swap(&mut updated_metadata, metadata);
    }

    fn register_strides<T: ToPrimitive>(
        strides: &[T],
        shape: &[T],
        old_rank: Option<u32>,
        output: &mut Vec<u32>,
    ) {
        let old_rank = if let Some(old_rank) = old_rank {
            let rank = output[0];
            let rank_diff = old_rank - rank;
            let padded_strides = if rank_diff > 0 {
                shape
                    .iter()
                    .take(old_rank as usize)
                    .map(|a| a.to_u32().unwrap())
                    .sum::<u32>()
            } else {
                0
            };

            for _ in 0..rank_diff {
                output.push(padded_strides.to_u32().unwrap());
            }

            old_rank as usize
        } else {
            output[0] as usize // same as current.
        };

        for stride in strides.iter().take(old_rank) {
            output.push(stride.to_u32().unwrap());
        }
    }

    fn register_shape<T: ToPrimitive>(shape: &[T], old_rank: Option<u32>, output: &mut Vec<u32>) {
        let old_rank = if let Some(old_rank) = old_rank {
            let rank = output[0];
            let rank_diff = rank - old_rank;

            for _ in 0..rank_diff {
                output.push(1);
            }

            old_rank as usize
        } else {
            output[0] as usize // same as current
        };

        for elem in shape.iter().take(old_rank) {
            output.push(elem.to_u32().unwrap());
        }
    }

    fn register(
        self,
        client: &ComputeClient<R::Server, R::Channel>,
        bindings_global: &mut Vec<Binding<R::Server>>,
    ) {
        if let Self::Some {
            bindings,
            mut metadata,
            lengths,
        } = self
        {
            if R::require_array_lengths() {
                for len in lengths {
                    metadata.push(len);
                }
            }

            bindings_global.extend(bindings);
            bindings_global.push(client.create(bytemuck::cast_slice(&metadata)).binding());
        }
    }
}

impl<T: NoUninit> ScalarState<T> {
    /// Add a new scalar value to the state.
    pub fn push(&mut self, val: T) {
        match self {
            ScalarState::Empty => *self = Self::Some(vec![val]),
            ScalarState::Some(values) => values.push(val),
        }
    }

    fn register<R: Runtime>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
        bindings: &mut Vec<Binding<R::Server>>,
    ) {
        match self {
            ScalarState::Empty => (),
            ScalarState::Some(values) => {
                let handle = client.create(bytemuck::cast_slice(values));
                bindings.push(handle.binding());
            }
        }
    }
}

impl<R: Runtime> Default for KernelLauncher<R> {
    fn default() -> Self {
        Self {
            tensors: TensorState::Empty,
            scalar_bf16: ScalarState::Empty,
            scalar_f16: ScalarState::Empty,
            scalar_f32: ScalarState::Empty,
            scalar_f64: ScalarState::Empty,
            scalar_u32: ScalarState::Empty,
            scalar_i64: ScalarState::Empty,
            scalar_i32: ScalarState::Empty,
            scalar_order: Vec::new(),
        }
    }
}
