use burn_core::{
    module::{AutodiffModule, Module},
    prelude::Backend,
};

use crate::{
    Learner, LearnerDataLoaders, SingleDeviceTrainEpoch, SingleDeviceValidEpoch, TrainLoader,
    TrainStep, ValidLoader, ValidStep, components::LearnerComponents,
    metric::processor::EventProcessor,
};

pub(crate) fn prepare_dataloaders_single_device<LC, TI, VI>(
    device: &<LC::Backend as Backend>::Device,
    dataloader_train: TrainLoader<LC, TI>,
    dataloader_valid: ValidLoader<LC, VI>,
) -> LearnerDataLoaders<LC, TI, VI>
where
    LC: LearnerComponents,
{
    // The reference model is always on the first device provided.
    let train = dataloader_train.to_device(device);
    let valid = dataloader_valid.to_device(device);

    LearnerDataLoaders::SingleTrainSingleValid {
        dataloader_train: train,
        dataloader_valid: valid,
    }
}

pub(crate) fn prepare_model_single_device<LC: LearnerComponents>(
    device: &<LC::Backend as Backend>::Device,
    mut learner: Learner<LC>,
) -> Learner<LC> {
    learner.model = learner.model.fork(device);

    learner
}

pub(crate) fn learn_single_device<LC, TI, VI, TO, VO>(
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
    let LearnerDataLoaders::SingleTrainSingleValid {
        dataloader_train,
        dataloader_valid,
    } = dataloaders
    else {
        panic!("Wrong dataloaders for strategy");
    };

    let mut epoch_train = SingleDeviceTrainEpoch::new(
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
