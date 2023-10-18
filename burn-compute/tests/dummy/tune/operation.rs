use burn_compute::tune::AutotuneKernel;
use derive_new::new;

use crate::dummy::{DummyElementwiseAddition, DummyKernel, DummyServer};

use super::DummyElementwiseAdditionSlowWrong;

// #[derive(new, PartialEq, Eq, Hash)]
// pub struct ArraysResource {
//     pub sizes: [usize; 3],
// }

// impl HashableResources for ArraysResource {
//     fn key(&self) -> String {
//         let mut hash = String::new();
//         for size in self.sizes {
//             let exp = f32::ceil(f32::log2(size as f32)) as u32;
//             hash.push_str(2_u32.pow(exp).to_string().as_str());
//             hash.push(',');
//         }
//         hash
//     }
// }

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
//
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
//

pub struct AdditionAutotuneKernel {}

impl AutotuneKernel<DummyServer> for AdditionAutotuneKernel {
    fn autotune_key(&self) -> String {
        todo!()
    }

    fn autotune_kernels(
        &self,
    ) -> Vec<<DummyServer as burn_compute::server::ComputeServer>::Kernel> {
        vec![
            Box::new(DummyElementwiseAddition),
            Box::new(DummyElementwiseAdditionSlowWrong),
        ]
    }

    fn autotune_handles(&self) -> &[&burn_compute::server::Handle<DummyServer>] {
        todo!()
    }

    fn fastest_kernel(
        &self,
        fastest_kernel_index: usize,
    ) -> <DummyServer as burn_compute::server::ComputeServer>::Kernel {
        todo!()
    }
}

pub fn get_addition_autotune_kernel() -> AdditionAutotuneKernel {
    AdditionAutotuneKernel {}
}
