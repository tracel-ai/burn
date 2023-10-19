use std::sync::Arc;

use burn_compute::tune::{AutotuneOperation, Operation};
use derive_new::new;

use crate::dummy::{
    DummyElementwiseAddition, DummyElementwiseMultiplication,
    DummyElementwiseMultiplicationSlowWrong, DummyKernel, DummyServer,
};

use super::DummyElementwiseAdditionSlowWrong;

#[derive(new)]
pub struct AdditionAutotuneKernel {
    shapes: Vec<Vec<usize>>,
}

impl AutotuneOperation<DummyServer> for AdditionAutotuneKernel {
    fn operation_key(&self) -> String {
        "add".to_string()
    }

    fn input_key(&self) -> String {
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
        let x: Arc<Box<dyn DummyKernel>> = Arc::new(Box::new(DummyElementwiseAddition));
        let y: Arc<Box<dyn DummyKernel>> = Arc::new(Box::new(DummyElementwiseAdditionSlowWrong));
        vec![Operation::new(x, None), Operation::new(y, None)]
    }

    fn inputs(&self) -> Vec<Vec<u8>> {
        arbitrary_bytes(&self.shapes)
    }
}

#[derive(new)]
pub struct MultiplicationAutotuneKernel {
    shapes: Vec<Vec<usize>>,
}

impl AutotuneOperation<DummyServer> for MultiplicationAutotuneKernel {
    fn operation_key(&self) -> String {
        "mul".to_string()
    }

    fn input_key(&self) -> String {
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
        let x: Arc<Box<dyn DummyKernel>> =
            Arc::new(Box::new(DummyElementwiseMultiplicationSlowWrong));
        let y: Arc<Box<dyn DummyKernel>> = Arc::new(Box::new(DummyElementwiseMultiplication));
        vec![Operation::new(x, None), Operation::new(y, None)]
        // vec![Operation::new(y, None)]
    }

    fn inputs(&self) -> Vec<Vec<u8>> {
        arbitrary_bytes(&self.shapes)
    }
}

pub fn arbitrary_bytes(shapes: &Vec<Vec<usize>>) -> Vec<Vec<u8>> {
    const ARBITRARY_BYTE: u8 = 12; // small so that squared < 256
    let mut handles = Vec::with_capacity(shapes.len());
    for shape in shapes {
        let n_bytes: usize = shape.iter().product();
        let mut handle = vec![ARBITRARY_BYTE; n_bytes];
        handles.push(handle)
    }
    handles
}

pub fn arbitrary_float_as_bytes(shapes: &Vec<Vec<usize>>) -> Vec<Vec<u8>> {
    const ARBITRARY_FLOAT: f32 = 1.234;
    let bytes: [u8; 4] = unsafe { std::mem::transmute(ARBITRARY_FLOAT) };

    let mut handles = Vec::with_capacity(shapes.len());
    for shape in shapes {
        let n_floats: usize = shape.iter().product();
        let mut handle = Vec::with_capacity(n_floats * 4);
        for _ in 0..n_floats {
            for j in 0..4 {
                handle.push(bytes[j])
            }
        }
        handles.push(handle)
    }
    handles
}
