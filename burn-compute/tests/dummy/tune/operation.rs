use std::sync::Arc;

use burn_compute::tune::{AutotuneOperation, Operation};
use derive_new::new;

use crate::dummy::{
    CacheTestFastOn3, CacheTestSlowOn3, DummyElementwiseAddition, DummyElementwiseMultiplication,
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
        log_shape_input_key(&self.shapes)
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
        log_shape_input_key(&self.shapes)
    }

    fn autotunables(&self) -> Vec<Operation<DummyServer>> {
        let x: Arc<Box<dyn DummyKernel>> =
            Arc::new(Box::new(DummyElementwiseMultiplicationSlowWrong));
        let y: Arc<Box<dyn DummyKernel>> = Arc::new(Box::new(DummyElementwiseMultiplication));
        vec![Operation::new(x, None), Operation::new(y, None)]
    }

    fn inputs(&self) -> Vec<Vec<u8>> {
        arbitrary_bytes(&self.shapes)
    }
}

#[derive(new)]
pub struct CacheTestAutotuneKernel {
    shapes: Vec<Vec<usize>>,
}

impl AutotuneOperation<DummyServer> for CacheTestAutotuneKernel {
    fn operation_key(&self) -> String {
        "cache_test".to_string()
    }

    fn input_key(&self) -> String {
        log_shape_input_key(&self.shapes)
    }

    fn autotunables(&self) -> Vec<Operation<DummyServer>> {
        let x: Arc<Box<dyn DummyKernel>> = Arc::new(Box::new(CacheTestFastOn3));
        let y: Arc<Box<dyn DummyKernel>> = Arc::new(Box::new(CacheTestSlowOn3));
        vec![Operation::new(x, None), Operation::new(y, None)]
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
        let handle = vec![ARBITRARY_BYTE; n_bytes];
        handles.push(handle)
    }
    handles
}

pub fn log_shape_input_key(shapes: &Vec<Vec<usize>>) -> String {
    let mut hash = String::new();
    let lhs = &shapes[0];
    for size in lhs {
        let exp = f32::ceil(f32::log2(*size as f32)) as u32;
        hash.push_str(2_u32.pow(exp).to_string().as_str());
        hash.push(',');
    }
    hash
}
