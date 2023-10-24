use std::sync::Arc;

use burn_compute::{
    server::Handle,
    tune::{AutotuneKey, AutotuneOperation, Operation},
};

use crate::dummy::{
    CacheTestFastOn3, CacheTestSlowOn3, DummyElementwiseAddition, DummyElementwiseMultiplication,
    DummyElementwiseMultiplicationSlowWrong, DummyKernel, DummyServer, ParameteredKernel,
};

use super::DummyElementwiseAdditionSlowWrong;

pub struct AdditionAutotuneKernel {
    key: AutotuneKey,
    shapes: Vec<Vec<usize>>,
}

impl AdditionAutotuneKernel {
    pub fn new(shapes: Vec<Vec<usize>>) -> Self {
        Self {
            key: AutotuneKey::new("add".to_string(), log_shape_input_key(&shapes)),
            shapes,
        }
    }
}

impl AutotuneOperation<DummyServer> for AdditionAutotuneKernel {
    fn key(&self) -> AutotuneKey {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<Operation<DummyServer>> {
        let x: Arc<dyn DummyKernel> = Arc::new(DummyElementwiseAddition);
        let y: Arc<dyn DummyKernel> = Arc::new(DummyElementwiseAdditionSlowWrong);
        vec![Operation::new(x, None), Operation::new(y, None)]
    }

    fn inputs(&self) -> Vec<Vec<u8>> {
        arbitrary_bytes(&self.shapes)
    }

    fn fastest(&self, fastest_index: usize) -> Operation<DummyServer> {
        self.autotunables()[fastest_index].clone()
    }
}

pub struct MultiplicationAutotuneKernel {
    key: AutotuneKey,
    shapes: Vec<Vec<usize>>,
}

impl MultiplicationAutotuneKernel {
    pub fn new(shapes: Vec<Vec<usize>>) -> Self {
        Self {
            key: AutotuneKey::new("mul".to_string(), log_shape_input_key(&shapes)),
            shapes,
        }
    }
}
impl AutotuneOperation<DummyServer> for MultiplicationAutotuneKernel {
    fn key(&self) -> AutotuneKey {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<Operation<DummyServer>> {
        let x: Arc<dyn DummyKernel> = Arc::new(DummyElementwiseMultiplicationSlowWrong);
        let y: Arc<dyn DummyKernel> = Arc::new(DummyElementwiseMultiplication);
        vec![Operation::new(x, None), Operation::new(y, None)]
    }

    fn inputs(&self) -> Vec<Vec<u8>> {
        arbitrary_bytes(&self.shapes)
    }

    fn fastest(&self, fastest_index: usize) -> Operation<DummyServer> {
        self.autotunables()[fastest_index].clone()
    }
}

pub struct CacheTestAutotuneKernel {
    key: AutotuneKey,
    shapes: Vec<Vec<usize>>,
}

impl CacheTestAutotuneKernel {
    pub fn new(shapes: Vec<Vec<usize>>) -> Self {
        Self {
            key: AutotuneKey::new("cache_test".to_string(), log_shape_input_key(&shapes)),
            shapes,
        }
    }
}
impl AutotuneOperation<DummyServer> for CacheTestAutotuneKernel {
    fn key(&self) -> AutotuneKey {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<Operation<DummyServer>> {
        let x: Arc<dyn DummyKernel> = Arc::new(CacheTestFastOn3);
        let y: Arc<dyn DummyKernel> = Arc::new(CacheTestSlowOn3);
        vec![Operation::new(x, None), Operation::new(y, None)]
    }

    fn inputs(&self) -> Vec<Vec<u8>> {
        arbitrary_bytes(&self.shapes)
    }

    fn fastest(&self, fastest_index: usize) -> Operation<DummyServer> {
        self.autotunables()[fastest_index].clone()
    }
}

pub struct ParameterTestAutotuneKernel {
    key: AutotuneKey,
    shapes: Vec<Vec<usize>>,
    parameter_handle: Handle<DummyServer>,
}

impl ParameterTestAutotuneKernel {
    pub fn new(shapes: Vec<Vec<usize>>, parameter_handle: Handle<DummyServer>) -> Self {
        Self {
            key: AutotuneKey::new("parameter_test".to_string(), log_shape_input_key(&shapes)),
            shapes,
            parameter_handle,
        }
    }
}
impl AutotuneOperation<DummyServer> for ParameterTestAutotuneKernel {
    fn key(&self) -> AutotuneKey {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<Operation<DummyServer>> {
        let x: Arc<dyn DummyKernel> = Arc::new(ParameteredKernel);
        let y: Arc<dyn DummyKernel> = Arc::new(DummyElementwiseAdditionSlowWrong);
        vec![
            Operation::new(x, Some(vec![self.parameter_handle.clone()])),
            Operation::new(y, None),
        ]
    }

    fn inputs(&self) -> Vec<Vec<u8>> {
        arbitrary_bytes(&self.shapes)
    }

    fn fastest(&self, fastest_index: usize) -> Operation<DummyServer> {
        self.autotunables()[fastest_index].clone()
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
