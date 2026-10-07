use alloc::vec::Vec;
use burn_backend::{
    TensorData,
    backend::ExecutionError,
    ops::{TransactionOps, TransactionPrimitive, TransactionPrimitiveData},
};
use burn_std::future::DynFut;
use core::future::Future;

use crate::{BackendRouter, RouterChannel, RouterClient, RouterTensor};

impl<R: RouterChannel> TransactionOps<Self> for BackendRouter<R> {
    fn tr_execute(
        transaction: TransactionPrimitive<Self>,
    ) -> impl Future<Output = Result<TransactionPrimitiveData, ExecutionError>> + Send {
        let floats = transaction.read_floats.len();
        let ints = transaction.read_ints.len();
        let reads = transaction.read_qfloats.is_empty().then(|| {
            TransactionReads::new(
                transaction
                    .read_floats
                    .into_iter()
                    .chain(transaction.read_ints)
                    .chain(transaction.read_bools),
            )
        });

        async move {
            let reads = reads.ok_or_else(|| {
                ExecutionError::generic("A router transaction cannot read quantized tensors yet")
            })?;
            let mut data = reads.wait().await?.into_iter();
            Ok(TransactionPrimitiveData {
                read_floats: data.by_ref().take(floats).collect(),
                read_qfloats: Vec::new(),
                read_ints: data.by_ref().take(ints).collect(),
                read_bools: data.collect(),
            })
        }
    }
}

/// One request per device, issued at construction.
struct TransactionReads(Vec<DeviceRead>);

struct DeviceRead {
    read: DynFut<Result<Vec<TensorData>, ExecutionError>>,
    positions: Vec<usize>,
}

struct DeviceTensors<C: RouterClient> {
    client: C,
    device: C::Device,
    tensors: Vec<RouterTensor<C>>,
    positions: Vec<usize>,
}

impl TransactionReads {
    fn new<C: RouterClient>(tensors: impl Iterator<Item = RouterTensor<C>>) -> Self {
        let mut devices: Vec<DeviceTensors<C>> = Vec::new();
        for (position, tensor) in tensors.enumerate() {
            let device = tensor.client.device();
            let index = match devices.iter().position(|tensors| tensors.device == device) {
                Some(index) => index,
                None => {
                    devices.push(DeviceTensors {
                        client: tensor.client.clone(),
                        device,
                        tensors: Vec::new(),
                        positions: Vec::new(),
                    });
                    devices.len() - 1
                }
            };
            devices[index].positions.push(position);
            devices[index].tensors.push(tensor);
        }
        Self(devices.into_iter().map(DeviceTensors::read).collect())
    }

    async fn wait(self) -> Result<Vec<TensorData>, ExecutionError> {
        let len = self.0.iter().map(|read| read.positions.len()).sum();
        let mut data: Vec<Option<TensorData>> =
            core::iter::repeat_with(|| None).take(len).collect();
        for device in self.0 {
            let values = device.read.await?;
            assert_eq!(
                values.len(),
                device.positions.len(),
                "A device answered a different number of values than tensors read"
            );
            for (position, value) in device.positions.into_iter().zip(values) {
                data[position] = Some(value);
            }
        }
        Ok(data
            .into_iter()
            .map(|value| value.expect("Each position belongs to one device"))
            .collect())
    }
}

impl<C: RouterClient> DeviceTensors<C> {
    fn read(self) -> DeviceRead {
        let tensors = self
            .tensors
            .into_iter()
            .map(RouterTensor::into_ir)
            .collect();
        DeviceRead {
            read: self.client.read_tensors_async(tensors),
            positions: self.positions,
        }
    }
}
