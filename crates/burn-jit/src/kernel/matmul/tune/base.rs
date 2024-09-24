use burn_tensor::{Element, ElementConversion};
use cubecl::tune::{local_tuner, AutotuneOperation, AutotuneOperationSet, LocalTuner};

use crate::{
    element::FloatElement,
    kernel::{matmul::utils::init_matmul_output, prng::random_like_uniform},
    ops::numeric::empty_device,
    tensor::JitTensor,
    tune_key::JitAutotuneKey,
    JitRuntime, JitTuneId,
};

use super::key::MatmulAutotuneKey;

/// Set of matmul implementations available for autotune
/// Autotune key is given by concatenating the closest upper power of 2 of m, k and n
pub struct MatmulAutotuneOperationSet<R: JitRuntime, E: FloatElement> {
    key: JitAutotuneKey,
    lhs: JitTensor<R, E>,
    rhs: JitTensor<R, E>,
    out: JitTensor<R, E>,
}
impl<R: JitRuntime, E: FloatElement> MatmulAutotuneOperationSet<R, E> {
    fn new(lhs: JitTensor<R, E>, rhs: JitTensor<R, E>, out: JitTensor<R, E>) -> Self {
        Self {
            key: JitAutotuneKey::Matmul(MatmulAutotuneKey::new(&lhs.shape, &rhs.shape)),
            lhs,
            rhs,
            out,
        }
    }
}

impl<R: JitRuntime, E: FloatElement> AutotuneOperationSet<JitAutotuneKey>
    for MatmulAutotuneOperationSet<R, E>
{
    fn key(&self) -> JitAutotuneKey {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<Box<dyn AutotuneOperation>> {
        let random_bounds: (E, E) = ((-10.0).elem::<E>(), (10.0).elem::<E>());
        let lhs = random_like_uniform(&self.lhs, random_bounds.0, random_bounds.1);
        let rhs = random_like_uniform(&self.rhs, random_bounds.0, random_bounds.1);

        let out = empty_device(
            self.out.client.clone(),
            self.out.device.clone(),
            self.out.shape.clone(),
        );

        vec![
            Box::new(SimpleMatmul::new(lhs.clone(), rhs.clone(), out.clone())),
            Box::new(SimpleMatmul16x16::new(
                lhs.clone(),
                rhs.clone(),
                out.clone(),
            )),
            Box::new(MatmulCube::new(lhs.clone(), rhs.clone(), out.clone())),
        ]
    }

    fn fastest(self: Box<Self>, fastest_index: usize) -> Box<dyn AutotuneOperation> {
        match fastest_index {
            0 => Box::new(SimpleMatmul::new(self.lhs, self.rhs, self.out)),
            1 => Box::new(SimpleMatmul16x16::new(self.lhs, self.rhs, self.out)),
            2 => Box::new(MatmulCube::new(self.lhs, self.rhs, self.out)),
            _ => panic!("Fastest index is out of bound"),
        }
    }
}

/// Executes autotune on matmul operations
pub fn matmul_autotune<R: JitRuntime, E: FloatElement + Element>(
    lhs: JitTensor<R, E>,
    rhs: JitTensor<R, E>,
) -> JitTensor<R, E> {
    let client = lhs.client.clone();

    let output = init_matmul_output(&lhs, &rhs);

    static TUNER: LocalTuner<JitAutotuneKey, JitTuneId> = local_tuner!();

    TUNER.execute(
        &JitTuneId::new::<R>(&lhs.device),
        &client,
        Box::new(MatmulAutotuneOperationSet::new(lhs, rhs, output.clone())),
    );

    output
}

macro_rules! matmul_tune_ops {
    ($name:ident, $func:expr) => {
        #[derive(new)]
        pub(crate) struct $name<R: JitRuntime, E: FloatElement> {
            lhs: JitTensor<R, E>,
            rhs: JitTensor<R, E>,
            out: JitTensor<R, E>,
        }

        impl<R: JitRuntime, E: FloatElement> AutotuneOperation for $name<R, E> {
            fn execute(self: Box<Self>) {
                #[allow(clippy::redundant_closure_call)]
                $func(self.lhs, self.rhs, self.out);
            }

            fn clone(&self) -> Box<dyn AutotuneOperation> {
                Box::new(Self {
                    lhs: self.lhs.clone(),
                    rhs: self.rhs.clone(),
                    out: self.out.clone(),
                })
            }
        }
    };
}

// Potentially better for small matrices.
matmul_tune_ops!(
    SimpleMatmul,
    crate::kernel::matmul::matmul_mem_coalescing_default
);

// Potentially better for small matrices.
matmul_tune_ops!(SimpleMatmul16x16, |lhs, rhs, out| {
    crate::kernel::matmul::matmul_simple(lhs, rhs, out, 16, 16)
});

// Probably the fastest in the general case, without loop unrolling
matmul_tune_ops!(
    MatmulCube,
    |lhs: JitTensor<R, E>, rhs: JitTensor<R, E>, out: JitTensor<R, E>| {
        cubecl::linalg::matmul::launch_ref::<R, E>(
            &lhs.client,
            lhs.as_handle_ref(),
            rhs.as_handle_ref(),
            out.as_handle_ref(),
        );
    }
);
