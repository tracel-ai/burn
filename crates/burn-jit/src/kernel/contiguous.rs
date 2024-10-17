use crate::{tensor::JitTensor, JitElement, JitRuntime};

/// Autotune types and functions for into_contiguous
#[cfg(feature = "autotune")]
#[allow(missing_docs)]
pub mod tune {
    use burn_tensor::{ElementConversion, Shape};
    use cubecl::{
        linalg::tensor::into_contiguous_prefetch,
        tensor_line_size, tune,
        tune::{local_tuner, tune_with, LocalTuner},
        AutotuneKey,
    };
    use rand::{seq::SliceRandom, thread_rng};
    use serde::{Deserialize, Serialize};

    use crate::{
        kernel::prng::random_uniform, ops::permute, tensor::JitTensor, JitAutotuneKey, JitElement,
        JitRuntime, JitTuneId,
    };

    /// Autotune key for [`into_contiguous`]
    #[derive(AutotuneKey, Serialize, Deserialize, PartialEq, Eq, Hash, Debug, Clone, Copy)]
    pub struct ContiguousKey {
        /// Total size of the tensor
        #[autotune(anchor)]
        pub size: usize,
        /// Rank of the tensor
        pub rank: usize,
        /// Vectorization of the tensor
        pub vectorization: u8,
    }

    /// Operations for [`into_contiguous`]
    #[tune(
        operations(prefetch_16, prefetch_8, prefetch_4, prefetch_2, prefetch_1),
        create_key = create_key,
    )]
    pub fn into_contiguous_operations<R: JitRuntime, E: JitElement>(
        key: JitAutotuneKey,
        tensor: JitTensor<R, E>,
    ) -> JitTensor<R, E> {
        let device = &tensor.device;
        let key = match key {
            JitAutotuneKey::Contiguous(key) => key,
            _ => unreachable!(),
        };

        let random_bounds: (E, E) = ((-1.0).elem::<E>(), (1.0).elem::<E>());
        let rank = key.rank;
        let num_elems = key.size;
        let mut dims = Vec::with_capacity(rank);
        for _ in 0..rank - 1 {
            let num_elems = f32::powf(num_elems as f32, 1.0 / rank as f32).floor();
            dims.push(num_elems as usize)
        }
        let last_dim = num_elems / dims.iter().product::<usize>();
        dims.push(last_dim - (last_dim % key.vectorization as usize));
        let tensor_shape = Shape { dims };
        let tensor = random_uniform(tensor_shape, device, random_bounds.0, random_bounds.1);
        let mut permute_axes: Vec<usize> = (0..rank).collect();
        permute_axes.shuffle(&mut thread_rng());

        tune_with!(permute(tensor, &permute_axes))
    }

    fn create_key<R: JitRuntime, E: JitElement>(tensor: &JitTensor<R, E>) -> JitAutotuneKey {
        let size = tensor.shape.num_elements();
        let rank = tensor.strides.len();
        let vectorization = tensor_line_size(
            R::supported_line_sizes(),
            &tensor.shape.dims,
            &tensor.strides,
            rank - 1,
        );
        JitAutotuneKey::Contiguous(ContiguousKey::new(size, rank, vectorization))
    }

    macro_rules! prefetch {
        ($name:ident, $num:expr) => {
            fn $name<R: JitRuntime, E: JitElement>(tensor: JitTensor<R, E>) -> JitTensor<R, E> {
                if tensor.is_contiguous() {
                    return tensor;
                }

                let output =
                    into_contiguous_prefetch::<R, E>(&tensor.client, tensor.as_handle_ref(), $num);

                JitTensor::new(
                    tensor.client,
                    output.handle,
                    output.shape.into(),
                    tensor.device,
                    output.strides,
                )
            }
        };
    }

    prefetch!(prefetch_16, 16);
    prefetch!(prefetch_8, 8);
    prefetch!(prefetch_4, 4);
    prefetch!(prefetch_2, 2);
    prefetch!(prefetch_1, 1);

    /// Executes autotune on conv2d operations
    pub fn into_contiguous_autotune<R: JitRuntime, E: JitElement>(
        tensor: JitTensor<R, E>,
    ) -> JitTensor<R, E> {
        let client = tensor.client.clone();

        static TUNER: LocalTuner<JitAutotuneKey, JitTuneId> = local_tuner!();

        TUNER.execute(
            &JitTuneId::new::<R>(&tensor.device),
            &client,
            Box::new(IntoContiguousOperations::<R, E>::new(tensor)),
        )
    }
}

/// Make a jit tensor contiguous.
pub fn into_contiguous<R: JitRuntime, E: JitElement>(tensor: JitTensor<R, E>) -> JitTensor<R, E> {
    if tensor.is_contiguous() {
        return tensor;
    }

    //#[cfg(not(feature = "autotune"))]
    {
        let output =
            cubecl::linalg::tensor::into_contiguous::<R, E>(&tensor.client, tensor.as_handle_ref());
        JitTensor::new(
            tensor.client,
            output.handle,
            output.shape.into(),
            tensor.device,
            output.strides,
        )
    }

    // #[cfg(feature = "autotune")]
    // tune::into_contiguous_autotune(tensor)
}
