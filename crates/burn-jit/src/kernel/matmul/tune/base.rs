use core::marker::PhantomData;

use burn_tensor::{Element, ElementConversion};
use cubecl::{
    linalg::matmul::{kernels::tiling2d::Tiling2dConfig, Strategy},
    tune::{local_tuner, AutotuneOperation, AutotuneOperationSet, LocalTuner},
};

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
    lhs: JitTensor<R>,
    rhs: JitTensor<R>,
    out: JitTensor<R>,
    _e: PhantomData<E>,
}
impl<R: JitRuntime, E: FloatElement> MatmulAutotuneOperationSet<R, E> {
    fn new(lhs: JitTensor<R>, rhs: JitTensor<R>, out: JitTensor<R>) -> Self {
        Self {
            key: JitAutotuneKey::Matmul(MatmulAutotuneKey::new(&lhs.shape, &rhs.shape, E::dtype())),
            lhs,
            rhs,
            out,
            _e: PhantomData,
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

        let out = empty_device::<R, E>(
            self.out.client.clone(),
            self.out.device.clone(),
            self.out.shape.clone(),
        );

        vec![
            Box::new(MatmulTiling2d::<R, E>::new(
                lhs.clone(),
                rhs.clone(),
                out.clone(),
            )),
            Box::new(MatmulAccelerated::<R, E>::new(
                lhs.clone(),
                rhs.clone(),
                out.clone(),
            )),
            Box::new(MatmulSimple::<R, E>::new(
                lhs.clone(),
                rhs.clone(),
                out.clone(),
            )),
        ]
    }

    fn fastest(self: Box<Self>, fastest_index: usize) -> Box<dyn AutotuneOperation> {
        match fastest_index {
            0 => Box::new(MatmulTiling2d::<R, E>::new(self.lhs, self.rhs, self.out)),
            1 => Box::new(MatmulAccelerated::<R, E>::new(self.lhs, self.rhs, self.out)),
            2 => Box::new(MatmulSimple::<R, E>::new(self.lhs, self.rhs, self.out)),
            _ => panic!("Fastest index is out of bound"),
        }
    }
}

/// Executes autotune on matmul operations
pub fn matmul_autotune<R: JitRuntime, E: FloatElement + Element>(
    lhs: JitTensor<R>,
    rhs: JitTensor<R>,
    out: Option<JitTensor<R>>,
) -> JitTensor<R> {
    let output = out.unwrap_or_else(|| init_matmul_output::<R, E>(&lhs, &rhs));

    let client = lhs.client.clone();

    static TUNER: LocalTuner<JitAutotuneKey, JitTuneId> = local_tuner!();

    TUNER.execute(
        &JitTuneId::new::<R>(&lhs.device),
        &client,
        Box::new(MatmulAutotuneOperationSet::<R, E>::new(
            lhs,
            rhs,
            output.clone(),
        )),
    );

    output
}

macro_rules! matmul_tune_ops {
    ($name:ident, $func:expr) => {
        #[derive(new, Debug)]
        pub(crate) struct $name<R: JitRuntime, E: FloatElement> {
            lhs: JitTensor<R>,
            rhs: JitTensor<R>,
            out: JitTensor<R>,
            _e: PhantomData<E>,
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
                    _e: self._e,
                })
            }
        }
    };
}

// Probably the fastest in the general case.
matmul_tune_ops!(
    MatmulAccelerated,
    |lhs: JitTensor<R>, rhs: JitTensor<R>, out: JitTensor<R>| {
        cubecl::linalg::matmul::launch_ref::<R, E>(
            &Strategy::Accelerated,
            &lhs.client,
            &lhs.as_handle_ref(),
            &rhs.as_handle_ref(),
            &out.as_handle_ref(),
        );
    }
);

// Probably the fastest when tensor cores are not available.
matmul_tune_ops!(
    MatmulTiling2d,
    |lhs: JitTensor<R>, rhs: JitTensor<R>, out: JitTensor<R>| {
        cubecl::linalg::matmul::launch_ref::<R, E>(
            &Strategy::Tiling2D(Tiling2dConfig::default()),
            &lhs.client,
            &lhs.as_handle_ref(),
            &rhs.as_handle_ref(),
            &out.as_handle_ref(),
        );
    }
);

// Probably the fastest for small matrices.
matmul_tune_ops!(
    MatmulSimple,
    |lhs: JitTensor<R>, rhs: JitTensor<R>, out: JitTensor<R>| {
        cubecl::linalg::matmul::launch_ref::<R, E>(
            &Strategy::Simple,
            &lhs.client,
            &lhs.as_handle_ref(),
            &rhs.as_handle_ref(),
            &out.as_handle_ref(),
        );
    }
);
