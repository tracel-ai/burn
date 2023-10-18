use burn_compute::{
    server::{ComputeServer, Handle},
    tune::{AutotuneOperation, Operation},
};
use derive_new::new;

use crate::dummy::{DummyElementwiseAddition, DummyKernel, DummyServer};

use super::DummyElementwiseAdditionSlowWrong;

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

    fn inputs(&self) -> Vec<Vec<u8>> {
        const ARBITRARY_BYTE: u8 = 42;
        let mut handles = Vec::with_capacity(self.shapes.len());
        for shape in &self.shapes {
            let n_bytes: usize = shape.iter().product();
            let handle = vec![ARBITRARY_BYTE; n_bytes];
            handles.push(handle)
        }
        handles
    }
}
