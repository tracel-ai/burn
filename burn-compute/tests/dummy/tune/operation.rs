use burn_compute::{
    channel::ComputeChannel,
    server::{ComputeServer, Handle},
    tune::{AutotuneOperation, Operation},
};
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

#[derive(new)]
pub struct AdditionAutotuneKernel {
    shapes: Vec<Vec<usize>>,
}

impl AutotuneOperation<DummyServer> for AdditionAutotuneKernel {
    fn key(&self) -> String {
        let mut hash = String::new();
        let lhs = &self.shapes[0];
        for size in lhs {
            let exp = f32::ceil(f32::log2(*size as f32)) as u32;
            hash.push_str(2_u32.pow(exp).to_string().as_str());
            hash.push(',');
        }
        hash
    }

    fn autotunables(&self) -> Vec<Operation<DummyServer>> {
        let x: Box<dyn DummyKernel> = Box::new(DummyElementwiseAddition);
        let y: Box<dyn DummyKernel> = Box::new(DummyElementwiseAdditionSlowWrong);
        vec![Operation::new(x, None), Operation::new(y, None)]
    }

    fn inputs(&self, server: &mut DummyServer) -> Vec<Handle<DummyServer>> {
        // todo!()
        // if cyclic mutable ref, return only list of bytes to create
        const ARBITRARY_BYTE: u8 = 42;
        let mut handles = Vec::with_capacity(self.shapes.len());
        for shape in &self.shapes {
            let n_elements: usize = shape.iter().product();
            let handle = server.create(&vec![ARBITRARY_BYTE; n_elements]);
            handles.push(handle)
        }
        handles
    }

    // fn autotune_key(&self) -> String {
    //     let mut hash = String::new();
    //     for size in self.shape.clone() {
    //         let exp = f32::ceil(f32::log2(size as f32)) as u32;
    //         hash.push_str(2_u32.pow(exp).to_string().as_str());
    //         hash.push(',');
    //     }
    //     hash
    // }

    //     fn autotune_kernels(
    //         &self,
    //     ) -> Vec<<DummyServer as burn_compute::server::ComputeServer>::Kernel> {
    //
    //     }

    //     fn autotune_handles(&self, server: &mut S) -> Vec<Handle<S>> {
    //
    //     }

    //     fn fastest_kernel(
    //         &self,
    //         fastest_kernel_index: usize,
    //     ) -> <DummyServer as burn_compute::server::ComputeServer>::Kernel {
    //         todo!()
    //     }
}
