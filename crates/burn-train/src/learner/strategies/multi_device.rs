use burn_core::{
    data::dataloader::split::split_dataloader,
    module::{AutodiffModule, Module},
    prelude::Backend,
};

use crate::{
    Learner, LearnerDataLoaders, MultiDeviceTrainEpoch, SingleDeviceValidEpoch, TrainLoader,
    TrainStep, ValidLoader, ValidStep, components::LearnerComponents,
    metric::processor::EventProcessor,
};

pub(crate) fn prepare_dataloaders_multi_device<LC, TI, VI>(
    devices: &[<LC::Backend as Backend>::Device],
    dataloader_train: TrainLoader<LC, TI>,
    dataloader_valid: ValidLoader<LC, VI>,
) -> LearnerDataLoaders<LC, TI, VI>
where
    LC: LearnerComponents,
{
    // `MultiDevicesTrainStep` has one worker per device, so we use a fixed device strategy
    // for each (worker) data loader. This matches the expected device on the worker, so we
    // don't have to move the data between devices.
    let train = split_dataloader(dataloader_train, devices);
    let main_device = devices.first().unwrap();
    let valid = dataloader_valid.to_device(main_device);

    LearnerDataLoaders::MultiTrainSingleValid {
        dataloader_train: train,
        dataloader_valid: valid,
    }
}

pub(crate) fn prepare_model_multi_device<LC: LearnerComponents>(
    devices: &[<LC::Backend as Backend>::Device],
    mut learner: Learner<LC>,
) -> Learner<LC> {
    let main_device = devices.first().unwrap();
    learner.model = learner.model.fork(main_device);

    learner
}

pub(crate) fn learn_multi_device<LC, TI, VI, TO, VO>(
    devices: &[<LC::Backend as Backend>::Device],
    mut learner: Learner<LC>,
    dataloaders: LearnerDataLoaders<LC, TI, VI>,
    starting_epoch: usize,
) -> Learner<LC>
where
    TI: Send + 'static,
    TO: Send + 'static,
    LC: LearnerComponents,
    LC::EventProcessor: EventProcessor<ItemValid = VO, ItemTrain = TO>,
    <LC::Model as AutodiffModule<LC::Backend>>::InnerModule: ValidStep<VI, VO>,
    <LC as LearnerComponents>::Model: TrainStep<TI, TO>,
{
    let LearnerDataLoaders::MultiTrainSingleValid {
        dataloader_train,
        dataloader_valid,
    } = dataloaders
    else {
        panic!("Wrong dataloaders for strategy");
    };

    let mut epoch_train = MultiDeviceTrainEpoch::new(
        dataloader_train,
        starting_epoch,
        learner.num_epochs,
        learner.grad_accumulation,
    );

    for epoch in starting_epoch..learner.num_epochs + 1 {
        (learner.model, learner.optim) = epoch_train.run::<LC, TO>(
            learner.model,
            learner.optim,
            &mut learner.lr_scheduler,
            &mut learner.event_processor,
            devices.to_vec(),
            &learner.interrupter,
        );

        if learner.interrupter.should_stop() {
            break;
        }

        let epoch_valid =
            SingleDeviceValidEpoch::new(dataloader_valid.clone(), epoch, learner.num_epochs);
        epoch_valid.run::<LC, VO>(
            &learner.model,
            &mut learner.event_processor,
            &learner.interrupter,
        );

        if let Some(checkpointer) = &mut learner.checkpointer {
            checkpointer.checkpoint(
                &learner.model,
                &learner.optim,
                &learner.lr_scheduler,
                epoch,
                &learner.event_store,
            );
        }

        if let Some(early_stopping) = &mut learner.early_stopping {
            if early_stopping.should_stop(epoch, &learner.event_store) {
                break;
            }
        }
    }

    learner
}
