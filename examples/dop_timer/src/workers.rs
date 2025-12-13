use crate::WorkRequest;
use burn::Tensor;
use burn::collective::{CollectiveConfig, PeerId, ReduceOperation, all_reduce};
use burn::prelude::Backend;
use burn::tensor::TensorPrimitive;
use std::sync::mpsc::Receiver;

struct Worker<B: Backend> {
    index: usize,
    id: PeerId,
    device: B::Device,
    config: CollectiveConfig,
}

impl<B: Backend> Worker<B> {
    pub fn new(index: usize, device: B::Device, config: CollectiveConfig) -> Self {
        let device = device.clone();
        let id = index.into();
        Self {
            index,
            id,
            device,
            config,
        }
    }

    #[tracing::instrument(skip(self, tensor))]
    pub fn dispatch_all_reduce<const R: usize>(
        &mut self,
        tensor: Tensor<B, R>,
        op: ReduceOperation,
    ) -> Tensor<B, R> {
        log::debug!("w={}: dispatch_all_reduce start", self.index);
        let tensor = Tensor::from_primitive(TensorPrimitive::Float(
            all_reduce::<B>(self.id, tensor.into_primitive().tensor(), op).unwrap(),
        ));
        log::debug!("w={}: dispatch_all_reduce end", self.index);
        tensor
    }

    pub fn run(&mut self, rx: Receiver<WorkRequest<B>>) {
        println!("worker {} started", self.index);
        while let Ok(command) = rx.recv() {
            use crate::WorkRequest::*;
            use burn::collective::register;
            match command {
                RegisterRequest { tx } => {
                    register::<B>(self.id, self.device.clone(), self.config.clone()).unwrap();
                    tx.send(()).unwrap();
                }
                AllReduceRequest { tensor, op, tx } => {
                    assert_eq!(&tensor.device(), &self.device);
                    let tensor = self.dispatch_all_reduce(tensor, op);
                    tx.send(tensor).unwrap();
                }
            }
        }
    }
}

pub struct WorkerHandle<B: Backend> {
    device: B::Device,
    tx: std::sync::mpsc::SyncSender<WorkRequest<B>>,
    phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> WorkerHandle<B> {
    #[tracing::instrument(skip(config))]
    pub fn new(index: usize, device: &B::Device, config: CollectiveConfig) -> Self {
        let type_id = 0;
        let mut worker: Worker<B> = Worker::new(index, device.clone(), config.clone());

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        std::thread::spawn(move || worker.run(rx));
        Self {
            device: device.clone(),
            tx,
            phantom: Default::default(),
        }
    }

    pub fn register(&self) -> Receiver<()> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        self.tx.send(WorkRequest::RegisterRequest { tx }).unwrap();
        rx
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }

    pub fn to_device<const R: usize>(&self, tensor: Tensor<B, R>) -> Tensor<B, R> {
        tensor.to_device(&self.device)
    }

    pub fn all_reduce(&self, op: ReduceOperation, tensor: Tensor<B, 4>) -> Receiver<Tensor<B, 4>> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        self.tx
            .send(WorkRequest::AllReduceRequest { tensor, op, tx })
            .unwrap();
        rx
    }
}
