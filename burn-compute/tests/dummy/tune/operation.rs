use burn_compute::tune::{AutotuneKernel, HashableResources};
use derive_new::new;

use crate::dummy::{DummyElementwiseAddition, DummyKernel, DummyServer};

#[derive(new, PartialEq, Eq, Hash)]
pub struct ArraysResource {
    pub sizes: [usize; 3],
}

impl HashableResources for ArraysResource {
    fn key(&self) -> String {
        let mut hash = String::new();
        for size in self.sizes {
            let exp = f32::ceil(f32::log2(size as f32)) as u32;
            hash.push_str(2_u32.pow(exp).to_string().as_str());
            hash.push(',');
        }
        hash
    }
}

// pub struct AdditionOp {}
// impl Operation for AdditionOp {
//     type Resources = ArraysResource;
// }

// #[derive(PartialEq, Eq, Hash)]
// pub struct MultiplicationOp {}
// impl Operation for MultiplicationOp {
//     type Resources = ArraysResource;
// }

// #[derive(PartialEq, Eq, Hash)]
// pub struct CacheTestOp {}
// impl Operation for CacheTestOp {
//     type Resources = ArraysResource;
// }

// macro_rules! make_operation {
//     ($name:ident, $resources:ident, [$($kernels:ident),*]) => {
//         impl Operation for $name {
//             type Resources = $resources
//         }
//     };
// }
// make_operation!(
//     AdditionOp,
//     ArraysResource,
//     [DummyElementwiseAddition, DummyElementwiseAdditionSlowWrong]
// );

// pub struct AdditionOp {
//     kernels: Vec<DummyKernel>,
// }

// impl Operation for AdditionOp {
//     type Resources = ArraysResource;

//     fn all_kernels(&self) -> Vec<S::Kernel> {
//         self.kernels
//     }
// }

pub struct AdditionAutotuneKernel {}

impl AutotuneKernel for AdditionAutotuneKernel {}

pub fn get_addition_autotune_kernel() -> AutotuneKernel {}
