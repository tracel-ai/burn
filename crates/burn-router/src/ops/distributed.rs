use alloc::vec::Vec;

use burn_backend::{
    DeviceId,
    distributed::{CollectiveTensor, DistributedOps, ReduceOperation},
    tensor::{Device, FloatTensor},
};
use burn_ir::{AllReduceOpIr, DeviceIdIr, DistributedOperationIr, OperationIr, OperationOutput};

use crate::{BackendRouter, RouterChannel, RouterClient, get_client};

impl<R: RouterChannel> DistributedOps<Self> for BackendRouter<R> {
    fn all_reduce(
        tensor: FloatTensor<Self>,
        op: ReduceOperation,
        device_ids: Vec<DeviceId>,
    ) -> CollectiveTensor<Self> {
        let client = tensor.client.clone();
        let device_ids = device_ids.into_iter().map(DeviceIdIr::from).collect();
        let desc = AllReduceOpIr::create(tensor.into_ir(), op, device_ids, || {
            client.create_empty_handle()
        });

        let output = client
            .register(OperationIr::Distributed(DistributedOperationIr::AllReduce(
                desc,
            )))
            .output();

        CollectiveTensor::new(output)
    }

    fn sync_collective(device: &Device<Self>) {
        // Fire-and-forget, like `all_reduce`: register a `SyncCollective` op on the device's
        // stream instead of a blocking call. The interpreter resolves it through the normal op
        // stream like any other distributed op.
        let client = get_client::<R>(device);
        client.register_op(OperationIr::Distributed(
            DistributedOperationIr::SyncCollective,
        ));
    }
}
