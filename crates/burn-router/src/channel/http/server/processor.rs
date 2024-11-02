use burn_tensor::{
    backend::{Backend, BackendBridge},
    repr::{ReprBackend, TensorDescription},
    TensorData,
};
use core::marker::PhantomData;
use std::sync::mpsc::Sender;

use crate::{
    http::{ReadTensor, RegisterOperation, RegisterOrphan, RegisterTensor, RegisterTensorEmpty},
    Runner, RunnerClient,
};

/// The goal of the processor is to asynchonously process compute tasks on it own thread.
pub struct Processor<B: ReprBackend> {
    p: PhantomData<B>,
}

pub type Callback<M> = Sender<M>;

pub enum ProcessorTask {
    RegisterOperation(RegisterOperation),
    RegisterTensor(RegisterTensor, Callback<TensorDescription>),
    RegisterTensorEmpty(RegisterTensorEmpty, Callback<TensorDescription>),
    ReadTensor(ReadTensor, Callback<TensorData>),
    Sync(Callback<()>),
    RegisterOrphan(RegisterOrphan),
    Close,
}

impl<B: ReprBackend> Processor<B>
where
    // Restrict full precision backend handle to be the same
    <<B as Backend>::FullPrecisionBridge as BackendBridge<B>>::Target:
        ReprBackend<Handle = B::Handle>,
{
    pub fn new(runner: Runner<B>) -> Sender<ProcessorTask> {
        let (sender, rec) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            for item in rec.iter() {
                match item {
                    ProcessorTask::RegisterOperation(val) => {
                        runner.register(val.op);
                    }
                    ProcessorTask::RegisterOrphan(val) => {
                        runner.register_orphan(&val.id);
                    }
                    ProcessorTask::Sync(callback) => {
                        runner.sync();
                        callback.send(()).unwrap();
                    }
                    ProcessorTask::RegisterTensor(val, callback) => {
                        let val = runner.register_tensor_data_desc(val.data);
                        callback.send(val).unwrap();
                    }
                    ProcessorTask::RegisterTensorEmpty(val, callback) => {
                        let val = runner.register_empty_tensor_desc(val.shape, val.dtype);
                        callback.send(val).unwrap();
                    }
                    ProcessorTask::ReadTensor(val, callback) => {
                        let tensor = burn_common::future::block_on(runner.read_tensor(val.tensor));
                        callback.send(tensor).unwrap();
                    }
                    ProcessorTask::Close => {
                        return;
                    }
                }
            }
        });

        sender
    }
}
