#[cfg(feature = "autotune-persistent-cache")]
use rand::{distributions::Alphanumeric, Rng};
use std::sync::Arc;

#[cfg(feature = "autotune-persistent-cache")]
use burn_compute::tune::compute_checksum;
use burn_compute::{
    server::Binding,
    tune::{AutotuneOperation, AutotuneOperationSet},
};

use crate::dummy::{
    CacheTestFastOn3, CacheTestSlowOn3, DummyClient, DummyElementwiseAddition,
    DummyElementwiseMultiplication, DummyElementwiseMultiplicationSlowWrong, DummyServer,
    OneKernelAutotuneOperation,
};

use super::DummyElementwiseAdditionSlowWrong;

pub struct AdditionAutotuneOperationSet {
    client: DummyClient,
    key: String,
    shapes: Vec<Vec<usize>>,
    bindings: Vec<Binding<DummyServer>>,
}

impl AdditionAutotuneOperationSet {
    #[allow(dead_code)]
    pub fn new(
        client: DummyClient,
        shapes: Vec<Vec<usize>>,
        bindings: Vec<Binding<DummyServer>>,
    ) -> Self {
        Self {
            client,
            key: format!("{}-{}", "add", log_shape_input_key(&shapes)),
            shapes,
            bindings,
        }
    }
}

impl AutotuneOperationSet<String> for AdditionAutotuneOperationSet {
    fn key(&self) -> String {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<Box<dyn AutotuneOperation>> {
        vec![
            Box::new(OneKernelAutotuneOperation::new(
                Arc::new(DummyElementwiseAddition),
                self.client.clone(),
                self.shapes.clone(),
                self.bindings.clone(),
            )),
            Box::new(OneKernelAutotuneOperation::new(
                Arc::new(DummyElementwiseAdditionSlowWrong),
                self.client.clone(),
                self.shapes.clone(),
                self.bindings.clone(),
            )),
        ]
    }

    fn fastest(self: Box<Self>, fastest_index: usize) -> Box<dyn AutotuneOperation> {
        self.autotunables()[fastest_index].clone()
    }
}

pub struct MultiplicationAutotuneOperationSet {
    client: DummyClient,
    key: String,
    shapes: Vec<Vec<usize>>,
    bindings: Vec<Binding<DummyServer>>,
}

impl MultiplicationAutotuneOperationSet {
    #[allow(dead_code)]
    pub fn new(
        client: DummyClient,
        shapes: Vec<Vec<usize>>,
        bindings: Vec<Binding<DummyServer>>,
    ) -> Self {
        Self {
            client,
            key: format!("{}-{}", "mul", log_shape_input_key(&shapes)),
            shapes,
            bindings,
        }
    }
}
impl AutotuneOperationSet<String> for MultiplicationAutotuneOperationSet {
    fn key(&self) -> String {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<Box<dyn AutotuneOperation>> {
        vec![
            Box::new(OneKernelAutotuneOperation::new(
                Arc::new(DummyElementwiseMultiplicationSlowWrong),
                self.client.clone(),
                self.shapes.clone(),
                self.bindings.clone(),
            )),
            Box::new(OneKernelAutotuneOperation::new(
                Arc::new(DummyElementwiseMultiplication),
                self.client.clone(),
                self.shapes.clone(),
                self.bindings.clone(),
            )),
        ]
    }

    fn fastest(self: Box<Self>, fastest_index: usize) -> Box<dyn AutotuneOperation> {
        self.autotunables()[fastest_index].clone()
    }
}

pub struct CacheTestAutotuneOperationSet {
    client: DummyClient,
    key: String,
    shapes: Vec<Vec<usize>>,
    bindings: Vec<Binding<DummyServer>>,
    pub generate_random_checksum: bool,
}

impl CacheTestAutotuneOperationSet {
    #[allow(dead_code)]
    pub fn new(
        client: DummyClient,
        shapes: Vec<Vec<usize>>,
        bindings: Vec<Binding<DummyServer>>,
    ) -> Self {
        Self {
            client,
            key: format!("{}-{}", "cache_test", log_shape_input_key(&shapes)),
            shapes,
            bindings,
            generate_random_checksum: false,
        }
    }
}

impl AutotuneOperationSet<String> for CacheTestAutotuneOperationSet {
    fn key(&self) -> String {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<Box<dyn AutotuneOperation>> {
        vec![
            Box::new(OneKernelAutotuneOperation::new(
                Arc::new(CacheTestFastOn3),
                self.client.clone(),
                self.shapes.clone(),
                self.bindings.clone(),
            )),
            Box::new(OneKernelAutotuneOperation::new(
                Arc::new(CacheTestSlowOn3),
                self.client.clone(),
                self.shapes.clone(),
                self.bindings.clone(),
            )),
        ]
    }

    fn fastest(self: Box<Self>, fastest_index: usize) -> Box<dyn AutotuneOperation> {
        self.autotunables()[fastest_index].clone()
    }

    #[cfg(feature = "std")]
    fn compute_checksum(&self) -> String {
        if self.generate_random_checksum {
            let rand_string: String = rand::thread_rng()
                .sample_iter(&Alphanumeric)
                .take(16)
                .map(char::from)
                .collect();
            rand_string
        } else {
            compute_checksum(&self.autotunables())
        }
    }
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
