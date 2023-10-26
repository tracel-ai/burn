use std::sync::Arc;

use burn_compute::{
    server::Handle,
    tune::{AutotuneKey, AutotuneOperation, AutotuneOperationSet},
};

use crate::dummy::{
    CacheTestFastOn3, CacheTestSlowOn3, DummyClient, DummyElementwiseAddition,
    DummyElementwiseMultiplication, DummyElementwiseMultiplicationSlowWrong, DummyServer,
    OneKernelAutotuneOperation,
};

use super::DummyElementwiseAdditionSlowWrong;

pub struct AdditionAutotuneOperationSet {
    client: DummyClient,
    key: AutotuneKey,
    shapes: Vec<Vec<usize>>,
    handles: Vec<Handle<DummyServer>>,
}

impl AdditionAutotuneOperationSet {
    pub fn new(
        client: DummyClient,
        shapes: Vec<Vec<usize>>,
        handles: Vec<Handle<DummyServer>>,
    ) -> Self {
        Self {
            client,
            key: AutotuneKey::new("add".to_string(), log_shape_input_key(&shapes)),
            shapes,
            handles,
        }
    }
}

impl AutotuneOperationSet<DummyServer> for AdditionAutotuneOperationSet {
    fn key(&self) -> AutotuneKey {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<Box<dyn AutotuneOperation<DummyServer>>> {
        vec![
            Box::new(OneKernelAutotuneOperation::new(
                Arc::new(DummyElementwiseAddition),
                self.client.clone(),
                self.shapes.clone(),
                self.handles.clone(),
            )),
            Box::new(OneKernelAutotuneOperation::new(
                Arc::new(DummyElementwiseAdditionSlowWrong),
                self.client.clone(),
                self.shapes.clone(),
                self.handles.clone(),
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
    handles: Vec<Handle<DummyServer>>,
}

impl<'a> MultiplicationAutotuneOperationSet {
    pub fn new(
        client: DummyClient,
        shapes: Vec<Vec<usize>>,
        handles: Vec<Handle<DummyServer>>,
    ) -> Self {
        Self {
            client,
            key: AutotuneKey::new("mul".to_string(), log_shape_input_key(&shapes)),
            shapes,
            handles,
        }
    }
}
impl AutotuneOperationSet<DummyServer> for MultiplicationAutotuneOperationSet {
    fn key(&self) -> AutotuneKey {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<Box<dyn AutotuneOperation<DummyServer>>> {
        vec![
            Box::new(OneKernelAutotuneOperation::new(
                Arc::new(DummyElementwiseMultiplicationSlowWrong),
                self.client.clone(),
                self.shapes.clone(),
                self.handles.clone(),
            )),
            Box::new(OneKernelAutotuneOperation::new(
                Arc::new(DummyElementwiseMultiplication),
                self.client.clone(),
                self.shapes.clone(),
                self.handles.clone(),
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
    handles: Vec<Handle<DummyServer>>,
}

impl CacheTestAutotuneOperationSet {
    pub fn new(
        client: DummyClient,
        shapes: Vec<Vec<usize>>,
        handles: Vec<Handle<DummyServer>>,
    ) -> Self {
        Self {
            client,
            key: AutotuneKey::new("cache_test".to_string(), log_shape_input_key(&shapes)),
            shapes,
            handles,
        }
    }
}
impl AutotuneOperationSet<DummyServer> for CacheTestAutotuneOperationSet {
    fn key(&self) -> AutotuneKey {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<Box<dyn AutotuneOperation<DummyServer>>> {
        vec![
            Box::new(OneKernelAutotuneOperation::new(
                Arc::new(CacheTestFastOn3),
                self.client.clone(),
                self.shapes.clone(),
                self.handles.clone(),
            )),
            Box::new(OneKernelAutotuneOperation::new(
                Arc::new(CacheTestSlowOn3),
                self.client.clone(),
                self.shapes.clone(),
                self.handles.clone(),
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
