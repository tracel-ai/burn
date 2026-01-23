use burn_core::tensor::Device;
use burn_rl::LearnerAgent;
use burn_rl::Policy;
use burn_rl::PolicyState;

use crate::RLAgentRecord;
use crate::{
    RLPolicyRecord, ReinforcementLearningComponentsTypes,
    checkpoint::Checkpointer,
    checkpoint::{AsyncCheckpointer, CheckpointingAction, CheckpointingStrategy},
    metric::store::EventStoreClient,
};

#[derive(new)]
/// Used to create, delete, or load checkpoints of the training process.
pub struct RLCheckpointer<RLC: ReinforcementLearningComponentsTypes> {
    policy: AsyncCheckpointer<RLPolicyRecord<RLC>, RLC::Backend>,
    learning_agent: AsyncCheckpointer<RLAgentRecord<RLC>, RLC::Backend>,
    strategy: Box<dyn CheckpointingStrategy>,
}

impl<RLC: ReinforcementLearningComponentsTypes> RLCheckpointer<RLC> {
    /// Create checkpoint for the training process.
    pub fn checkpoint(
        &mut self,
        policy: &RLC::Policy,
        learning_agent: &RLC::LearningAgent,
        epoch: usize,
        store: &EventStoreClient,
    ) {
        let actions = self.strategy.checkpointing(epoch, store);

        for action in actions {
            match action {
                CheckpointingAction::Delete(epoch) => {
                    self.policy
                        .delete(epoch)
                        .expect("Can delete policy checkpoint.");
                    self.learning_agent
                        .delete(epoch)
                        .expect("Can delete learning agent checkpoint.")
                }
                CheckpointingAction::Save => {
                    self.policy
                        .save(epoch, policy.state().into_record())
                        .expect("Can save policy checkpoint.");
                    self.learning_agent
                        .save(epoch, learning_agent.into_record())
                        .expect("Can save learning agent checkpoint.");
                }
            }
        }
    }

    /// Load a training checkpoint.
    pub fn load_checkpoint(
        &self,
        mut policy: RLC::Policy,
        mut learning_agent: RLC::LearningAgent,
        device: &Device<RLC::Backend>,
        epoch: usize,
    ) -> (RLC::Policy, RLC::LearningAgent) {
        let record = self
            .policy
            .restore(epoch, device)
            .expect("Can load model checkpoint.");
        let state = policy.state().load_record(record);
        policy.update(state);

        let record = self
            .learning_agent
            .restore(epoch, device)
            .expect("Can load learning agent checkpoint.");
        learning_agent = learning_agent.load_record(record);

        (policy, learning_agent)
    }
}
