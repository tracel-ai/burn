use derive_new::new;
use std::{
    marker::PhantomData,
    sync::mpsc::{self, Sender},
    thread::spawn,
};

use burn_core::{prelude::*, record::Record, tensor::backend::AutodiffBackend};

use crate::TransitionBatch;

/// An action along with additionnal context about the decision.
#[derive(Clone, new)]
pub struct ActionContext<A, C> {
    /// The context.
    pub context: C,
    /// The action.
    pub action: A,
}

/// The state of a policy.
pub trait PolicyState<B: Backend> {
    /// The type of the record.
    type Record: Record<B>;

    /// Convert the state to a record.
    fn into_record(&self) -> Self::Record;
    /// Load the state from a record.
    fn load_record(&self, record: Self::Record) -> Self;
}

/// Trait for a RL policy.
pub trait Policy<B: Backend>: Clone {
    /// The observation given as input to the policy.
    type Observation;
    /// The action distribution parameters defining how the action will be sampled.
    type ActionDistribution;
    /// The action.
    type Action;

    /// Additionnal context on the policy's decision.
    type ActionContext;
    /// The current parameterization of the policy.
    type PolicyState: PolicyState<B>;

    /// Produces the action distribution from a batch of observations.
    fn forward(&mut self, obs: Self::Observation) -> Self::ActionDistribution;
    /// Gives the action from a batch of observations.
    fn action(
        &mut self,
        obs: Self::Observation,
        deterministic: bool,
    ) -> (Self::Action, Vec<Self::ActionContext>);

    /// Update the policy's parameters.
    fn update(&mut self, update: Self::PolicyState);
    /// Returns the current parameterization.
    fn state(&self) -> Self::PolicyState;

    /// Loads the policy parameters from a record.
    fn from_record(&self, record: <Self::PolicyState as PolicyState<B>>::Record) -> Self;
}

/// Trait for a type that can be batched and unbatched (split).
pub trait Batchable: Sized {
    /// Create a batch from a list of items.
    fn batch(value: Vec<Self>) -> Self;
    /// Create a list from batched items.
    fn unbatch(self) -> Vec<Self>;
}

/// A training output.
pub struct RLTrainOutput<TO, P> {
    /// The policy.
    pub policy: P,
    /// The item.
    pub item: TO,
}

/// Learner for a policy.
pub trait PolicyLearner<B>
where
    B: AutodiffBackend,
    <Self::InnerPolicy as Policy<B>>::Observation: Clone + Batchable,
    <Self::InnerPolicy as Policy<B>>::ActionDistribution: Clone + Batchable,
    <Self::InnerPolicy as Policy<B>>::Action: Clone + Batchable,
{
    /// Additionnal context of a training step.
    type TrainContext;
    /// The policy to train.
    type InnerPolicy: Policy<B>;
    /// The record of the learner.
    type Record: Record<B>;

    /// Execute a training step on the policy.
    fn train(
        &mut self,
        input: TransitionBatch<
            B,
            <Self::InnerPolicy as Policy<B>>::Observation,
            <Self::InnerPolicy as Policy<B>>::Action,
        >,
    ) -> RLTrainOutput<Self::TrainContext, <Self::InnerPolicy as Policy<B>>::PolicyState>;
    /// Returns the learner's current policy.
    fn policy(&self) -> Self::InnerPolicy;
    /// Update the learner's policy.
    fn update_policy(&mut self, update: Self::InnerPolicy);

    /// Convert the learner's state into a record.
    fn into_record(&self) -> Self::Record;
    /// Load the learner's state from a record.
    fn load_record(self, record: Self::Record) -> Self;
}

#[derive(Clone)]
struct AutoBatcher<B: Backend, P: Policy<B>> {
    autobatch_size: usize,
    inner_policy: P,
    batch_action: Vec<ActionItem<P::Observation, P::Action, P::ActionContext>>,
    batch_logits: Vec<ForwardItem<P::Observation, P::ActionDistribution>>,
    _backend: PhantomData<B>,
}

impl<B, P> AutoBatcher<B, P>
where
    B: Backend,
    P: Policy<B>,
    P::Observation: Clone + Batchable,
    P::ActionDistribution: Clone + Batchable,
    P::Action: Clone + Batchable,
    P::ActionContext: Clone,
{
    pub fn new(autobatch_size: usize, inner_policy: P) -> Self {
        Self {
            autobatch_size,
            inner_policy,
            batch_action: vec![],
            batch_logits: vec![],
            _backend: PhantomData,
        }
    }

    pub fn push_action(&mut self, item: ActionItem<P::Observation, P::Action, P::ActionContext>) {
        self.batch_action.push(item);
        if self.len_actions() >= self.autobatch_size {
            self.flush_actions();
        }
    }

    pub fn push_logits(&mut self, item: ForwardItem<P::Observation, P::ActionDistribution>) {
        self.batch_logits.push(item);
        if self.len_logits() >= self.autobatch_size {
            self.flush_actions();
        }
    }

    pub fn len_actions(&self) -> usize {
        self.batch_action.len()
    }

    pub fn len_logits(&self) -> usize {
        self.batch_action.len()
    }

    pub fn flush_actions(&mut self) {
        let input: Vec<_> = self
            .batch_action
            .iter()
            .map(|m| m.inference_state.clone())
            .collect();
        // Only deterministic if all actions are requested as deterministic.
        let deterministic = self.batch_action.iter().all(|item| item.deterministic);
        let (actions, context) = self
            .inner_policy
            .action(P::Observation::batch(input), deterministic);
        let actions: Vec<_> = actions.unbatch();

        for (i, item) in self.batch_action.iter().enumerate() {
            item.sender
                .send(ActionContext {
                    context: vec![context[i].clone()],
                    action: actions[i].clone(),
                })
                .expect("Autobatcher should be able to send resulting actions.");
        }
        self.batch_action.clear();
    }

    pub fn flush_logits(&mut self) {
        let input: Vec<_> = self
            .batch_logits
            .iter()
            .map(|m| m.inference_state.clone())
            .collect();
        let output = self.inner_policy.forward(P::Observation::batch(input));
        let logits: Vec<_> = output.unbatch();
        for (i, item) in self.batch_logits.iter().enumerate() {
            item.sender
                .send(logits[i].clone())
                .expect("Autobatcher should be able to send resulting probabilities.");
        }
        self.batch_action.clear();
    }

    pub fn update_policy(&mut self, policy_update: P::PolicyState) {
        if self.len_actions() > 0 {
            self.flush_actions();
        }
        if self.len_logits() > 0 {
            self.flush_logits();
        }
        self.inner_policy.update(policy_update);
    }

    pub fn state(&self) -> P::PolicyState {
        self.inner_policy.state()
    }
}

enum InferenceMessage<B: Backend, P: Policy<B>> {
    ActionMessage(ActionItem<P::Observation, P::Action, P::ActionContext>),
    ForwardMessage(ForwardItem<P::Observation, P::ActionDistribution>),
    PolicyUpdate(P::PolicyState),
    PolicyRequest(Sender<P::PolicyState>),
}

#[derive(Clone)]
struct ActionItem<S, A, C> {
    sender: Sender<ActionContext<A, Vec<C>>>,
    inference_state: S,
    deterministic: bool,
}

#[derive(Clone)]
struct ForwardItem<S, O> {
    sender: Sender<O>,
    inference_state: S,
}

/// An asynchronous policy with autobatching for inference.
#[derive(Clone)]
pub struct AsyncPolicy<B: Backend, P: Policy<B>> {
    inference_state_sender: Sender<InferenceMessage<B, P>>,
}

impl<B, P> AsyncPolicy<B, P>
where
    B: Backend,
    P: Policy<B> + Clone + Send + 'static,
    P::ActionContext: Clone + Send,
    P::PolicyState: Send,
    P::Observation: Clone + Send + Batchable,
    P::ActionDistribution: Clone + Send + Batchable,
    P::Action: Clone + Send + Batchable,
{
    /// Create the policy.
    ///
    /// # Arguments
    ///
    /// * `autobatch_size` - Number of observation to accumulate before running a pass of inference.
    /// * `inner_policy` - The policy used to take actions.
    pub fn new(autobatch_size: usize, inner_policy: P) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();
        let mut autobatcher = AutoBatcher::new(autobatch_size, inner_policy.clone());
        spawn(move || {
            loop {
                match receiver.recv() {
                    Ok(msg) => match msg {
                        InferenceMessage::ActionMessage(item) => autobatcher.push_action(item),
                        InferenceMessage::ForwardMessage(item) => autobatcher.push_logits(item),
                        InferenceMessage::PolicyUpdate(update) => autobatcher.update_policy(update),
                        InferenceMessage::PolicyRequest(sender) => sender
                            .send(autobatcher.state())
                            .expect("Autobatcher should be able to send current policy state."),
                    },
                    Err(err) => {
                        log::error!("Error in AsyncPolicy : {}", err);
                        break;
                    }
                }
            }
        });

        Self {
            inference_state_sender: sender,
        }
    }
}

impl<B, P> Policy<B> for AsyncPolicy<B, P>
where
    B: Backend,
    P: Policy<B> + Send + 'static,
{
    type ActionContext = P::ActionContext;
    type PolicyState = P::PolicyState;

    type Observation = P::Observation;
    type ActionDistribution = P::ActionDistribution;
    type Action = P::Action;

    fn forward(&mut self, states: Self::Observation) -> Self::ActionDistribution {
        let (action_sender, action_receiver) = std::sync::mpsc::channel();
        let item = ForwardItem {
            sender: action_sender,
            inference_state: states,
        };
        self.inference_state_sender
            .send(InferenceMessage::ForwardMessage(item))
            .expect("Should be able to send message to inference_server");
        action_receiver
            .recv()
            .expect("AsyncPolicy should receive queued probabilities.")
    }

    fn action(
        &mut self,
        states: Self::Observation,
        deterministic: bool,
    ) -> (Self::Action, Vec<Self::ActionContext>) {
        let (action_sender, action_receiver) = std::sync::mpsc::channel();
        let item = ActionItem {
            sender: action_sender,
            inference_state: states,
            deterministic,
        };
        self.inference_state_sender
            .send(InferenceMessage::ActionMessage(item))
            .expect("should be able to send message to inference_server.");
        let action = action_receiver
            .recv()
            .expect("AsyncPolicy should receive queued actions.");
        (action.action, action.context)
    }

    fn update(&mut self, update: Self::PolicyState) {
        self.inference_state_sender
            .send(InferenceMessage::PolicyUpdate(update))
            .expect("AsyncPolicy should be able to send policy state.")
    }

    fn state(&self) -> Self::PolicyState {
        let (sender, receiver) = mpsc::channel();
        self.inference_state_sender
            .send(InferenceMessage::PolicyRequest(sender))
            .expect("should be able to send message to inference_server.");
        receiver
            .recv()
            .expect("AsyncPolicy should be able to receive policy state.")
    }

    fn from_record(&self, _record: <Self::PolicyState as PolicyState<B>>::Record) -> Self {
        // Not needed for now
        todo!()
    }
}
