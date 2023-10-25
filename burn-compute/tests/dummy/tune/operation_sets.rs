use std::sync::Arc;

use burn_compute::tune::{AutotuneKey, AutotuneOperation, AutotuneOperationSet};

use crate::dummy::{
    CacheTestFastOn3, CacheTestSlowOn3, DummyClient, DummyElementwiseAddition,
    DummyElementwiseMultiplication, DummyElementwiseMultiplicationSlowWrong, DummyKernel,
    DummyServer, OneKernelAutotuneOperation,
};

use super::DummyElementwiseAdditionSlowWrong;

pub struct AdditionAutotuneOperationSet {
    client: DummyClient,
    key: AutotuneKey,
    shapes: Vec<Vec<usize>>,
}

impl AdditionAutotuneOperationSet {
    pub fn new(client: DummyClient, shapes: Vec<Vec<usize>>) -> Self {
        Self {
            client,
            key: AutotuneKey::new("add".to_string(), log_shape_input_key(&shapes)),
            shapes,
        }
    }
}

impl AutotuneOperationSet<DummyServer> for AdditionAutotuneOperationSet {
    fn key(&self) -> AutotuneKey {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<Box<dyn AutotuneOperation<DummyServer>>> {
        let x: Arc<dyn DummyKernel> = Arc::new(DummyElementwiseAddition);
        let y: Arc<dyn DummyKernel> = Arc::new(DummyElementwiseAdditionSlowWrong);
        vec![
            Box::new(OneKernelAutotuneOperation::new(
                x,
                self.client.clone(),
                self.shapes.clone(),
            )),
            Box::new(OneKernelAutotuneOperation::new(
                y,
                self.client.clone(),
                self.shapes.clone(),
            )),
        ]
    }

    fn fastest(&self, fastest_index: usize) -> Box<dyn AutotuneOperation<DummyServer>> {
        self.autotunables()[fastest_index].clone()
    }
}

pub struct MultiplicationAutotuneOperationSet {
    client: DummyClient,
    key: AutotuneKey,
    shapes: Vec<Vec<usize>>,
}

impl<'a> MultiplicationAutotuneOperationSet {
    pub fn new(client: DummyClient, shapes: Vec<Vec<usize>>) -> Self {
        Self {
            client,
            key: AutotuneKey::new("mul".to_string(), log_shape_input_key(&shapes)),
            shapes,
        }
    }
}
impl AutotuneOperationSet<DummyServer> for MultiplicationAutotuneOperationSet {
    fn key(&self) -> AutotuneKey {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<Box<dyn AutotuneOperation<DummyServer>>> {
        let x: Arc<dyn DummyKernel> = Arc::new(DummyElementwiseMultiplicationSlowWrong);
        let y: Arc<dyn DummyKernel> = Arc::new(DummyElementwiseMultiplication);
        vec![
            Box::new(OneKernelAutotuneOperation::new(
                x,
                self.client.clone(),
                self.shapes.clone(),
            )),
            Box::new(OneKernelAutotuneOperation::new(
                y,
                self.client.clone(),
                self.shapes.clone(),
            )),
        ]
    }

    fn fastest(&self, fastest_index: usize) -> Box<dyn AutotuneOperation<DummyServer>> {
        self.autotunables()[fastest_index].clone()
    }
}

pub struct CacheTestAutotuneOperationSet {
    client: DummyClient,
    key: AutotuneKey,
    shapes: Vec<Vec<usize>>,
}

impl CacheTestAutotuneOperationSet {
    pub fn new(client: DummyClient, shapes: Vec<Vec<usize>>) -> Self {
        Self {
            client,
            key: AutotuneKey::new("cache_test".to_string(), log_shape_input_key(&shapes)),
            shapes,
        }
    }
}
impl AutotuneOperationSet<DummyServer> for CacheTestAutotuneOperationSet {
    fn key(&self) -> AutotuneKey {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<Box<dyn AutotuneOperation<DummyServer>>> {
        let x: Arc<dyn DummyKernel> = Arc::new(CacheTestFastOn3);
        let y: Arc<dyn DummyKernel> = Arc::new(CacheTestSlowOn3);
        vec![
            Box::new(OneKernelAutotuneOperation::new(
                x,
                self.client.clone(),
                self.shapes.clone(),
            )),
            Box::new(OneKernelAutotuneOperation::new(
                y,
                self.client.clone(),
                self.shapes.clone(),
            )),
        ]
    }

    fn fastest(&self, fastest_index: usize) -> Box<dyn AutotuneOperation<DummyServer>> {
        self.autotunables()[fastest_index].clone()
    }
}

// pub fn arbitrary_bytes(shapes: &Vec<Vec<usize>>) -> Vec<Vec<u8>> {
//     const ARBITRARY_BYTE: u8 = 12; // small so that squared < 256
//     let mut handles = Vec::with_capacity(shapes.len());
//     for shape in shapes {
//         let n_bytes: usize = shape.iter().product();
//         let handle = vec![ARBITRARY_BYTE; n_bytes];
//         handles.push(handle)
//     }
//     handles
// }

pub fn log_shape_input_key(shapes: &[Vec<usize>]) -> String {
    let mut hash = String::new();
    let lhs = &shapes[0];
    for size in lhs {
        let exp = f32::ceil(f32::log2(*size as f32)) as u32;
        hash.push_str(2_u32.pow(exp).to_string().as_str());
        hash.push(',');
    }
    hash
}
