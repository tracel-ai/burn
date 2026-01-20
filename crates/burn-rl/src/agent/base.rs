use derive_new::new;
use std::{marker::PhantomData, sync::mpsc::Sender, thread::spawn};

use burn_core::{prelude::*, tensor::backend::AutodiffBackend};

use crate::{Transition, TransitionBuffer};

#[derive(Clone, new)]
pub struct ActionContext<A, C> {
    pub context: C,
    pub action: A,
}

pub trait Policy<B: Backend, S, A>: Clone {
    type ActionContext: Clone;
    type PolicyState;

    fn logits(&mut self, state: &S) -> Tensor<B, 1>;
    fn action(&mut self, state: &S, deterministic: bool) -> ActionContext<A, Self::ActionContext>;

    fn batch_logits(&mut self, states: Vec<&S>) -> Tensor<B, 2>;
    fn batch_action(
        &mut self,
        states: Vec<&S>,
        deterministic: bool,
    ) -> Vec<ActionContext<A, Self::ActionContext>>;

    fn update(&mut self, update: Self::PolicyState);
    fn state(&self) -> Self::PolicyState;
}

pub trait OffPolicyTransitionBatch<B: AutodiffBackend> {
    fn from_samples(
        states_batch: Tensor<B, 2>,
        next_states_batch: Tensor<B, 2>,
        actions_batch: Tensor<B, 2>,
        rewards_batch: Tensor<B, 2>,
        dones_batch: Tensor<B, 2>,
    ) -> Self;
}

/// A training output.
pub struct RLTrainOutput<TO, P> {
    /// The policy.
    pub policy: P,

    /// The item.
    pub item: TO,
}

pub trait LearnerAgent<B: AutodiffBackend, S, A> {
    type TrainingInput: From<Transition<B>>;
    type TrainingOutput;
    type InnerPolicy: Policy<B, S, A>;

    fn train(
        &mut self,
        input: &TransitionBuffer<Self::TrainingInput>,
    ) -> RLTrainOutput<Self::TrainingOutput, <Self::InnerPolicy as Policy<B, S, A>>::PolicyState>;
    fn policy(&self) -> Self::InnerPolicy;
}

#[derive(Clone)]
struct AutoBatcher<B: Backend, S, A, P: Policy<B, S, A>> {
    autobatch_size: usize,
    inner_policy: P,
    batch_action: Vec<ActionItem<S, A, P::ActionContext>>,
    batch_logits: Vec<LogitsItem<B, S>>,
    _backend: PhantomData<B>,
}

impl<B: Backend, S, A, P: Policy<B, S, A>> AutoBatcher<B, S, A, P> {
    pub fn new(autobatch_size: usize, inner_policy: P) -> Self {
        Self {
            autobatch_size,
            inner_policy,
            batch_action: vec![],
            batch_logits: vec![],
            _backend: PhantomData,
        }
    }

    pub fn push_action(&mut self, item: ActionItem<S, A, P::ActionContext>) {
        self.batch_action.push(item);
        if self.len_actions() >= self.autobatch_size {
            self.flush_actions();
        }
    }

    pub fn push_logits(&mut self, item: LogitsItem<B, S>) {
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
        let input = self
            .batch_action
            .iter()
            .map(|m| &m.inference_state)
            .collect();
        // Only deterministic if all actions are requested as deterministic.
        let deterministic = self.batch_action.iter().all(|item| item.deterministic);
        let action_context = self.inner_policy.batch_action(input, deterministic);
        for (item, context) in self.batch_action.iter().zip(action_context.into_iter()) {
            item.sender.send(context).unwrap();
        }
        self.batch_action.clear();
    }

    pub fn flush_logits(&mut self) {
        let input = self
            .batch_logits
            .iter()
            .map(|m| &m.inference_state)
            .collect();
        let output = self.inner_policy.batch_logits(input);
        for (i, item) in self.batch_logits.iter().enumerate() {
            item.sender.send(output.clone().slice(s![i, ..])).unwrap();
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
}

enum InferenceMessage<B: Backend, S, A, P: Policy<B, S, A>> {
    ActionMessage(ActionItem<S, A, P::ActionContext>),
    LogitsMessage(LogitsItem<B, S>),
    Policy(P::PolicyState),
}

#[derive(Clone)]
struct ActionItem<S, A, C> {
    sender: Sender<ActionContext<A, C>>,
    inference_state: S,
    deterministic: bool,
}

#[derive(Clone)]
struct LogitsItem<B: Backend, S> {
    sender: Sender<Tensor<B, 2>>,
    inference_state: S,
}

#[derive(Clone)]
pub struct AsyncPolicy<B: Backend, S, A, P: Policy<B, S, A>> {
    inference_state_sender: Option<Sender<InferenceMessage<B, S, A, P>>>,
}

impl<B, S, A, P> AsyncPolicy<B, S, A, P>
where
    B: Backend,
    S: Send + 'static,
    A: Send + 'static,
    P: Policy<B, S, A> + Clone + Send + 'static,
    P::ActionContext: Send,
    P::PolicyState: Send,
{
    pub fn new(autobatch_size: usize, inner_policy: P) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();
        let mut autobatcher = AutoBatcher::new(autobatch_size, inner_policy.clone());
        // TODO: worker?
        spawn(move || {
            loop {
                match receiver
                    .recv()
                    .expect("Should be able to receive inference messages.")
                {
                    InferenceMessage::ActionMessage(item) => autobatcher.push_action(item),
                    InferenceMessage::LogitsMessage(item) => autobatcher.push_logits(item),
                    InferenceMessage::Policy(update) => autobatcher.update_policy(update),
                }
            }
        });

        Self {
            inference_state_sender: Some(sender),
        }
    }
}

impl<B, S, A, P> Policy<B, S, A> for AsyncPolicy<B, S, A, P>
where
    B: Backend,
    S: Clone,
    A: Clone,
    P: Policy<B, S, A> + Send + 'static,
{
    type ActionContext = P::ActionContext;
    type PolicyState = P::PolicyState;

    fn logits(&mut self, state: &S) -> Tensor<B, 1> {
        self.batch_logits(vec![state]).slice(s![0, ..]).squeeze()
    }

    fn action(&mut self, state: &S, deterministic: bool) -> ActionContext<A, Self::ActionContext> {
        self.batch_action(vec![state], deterministic).remove(0)
    }

    fn batch_logits(&mut self, states: Vec<&S>) -> Tensor<B, 2> {
        let (action_sender, action_receiver) = std::sync::mpsc::channel();
        // TODO: Don't Assume that only one state is passed.
        let item = LogitsItem {
            sender: action_sender,
            inference_state: states[0].clone(),
        };
        self.inference_state_sender
            .as_ref()
            .expect("Should call start() before queue_action().")
            .send(InferenceMessage::LogitsMessage(item))
            .expect("Should be able to send message to inference_server");
        // ec::from([action_receiver.recv().unwrap()])
        action_receiver.recv().unwrap()
    }

    fn batch_action(
        &mut self,
        state: Vec<&S>,
        deterministic: bool,
    ) -> Vec<ActionContext<A, Self::ActionContext>> {
        let (action_sender, action_receiver) = std::sync::mpsc::channel();
        // TODO: Don't Assume that only one state is passed.
        let item = ActionItem {
            sender: action_sender,
            inference_state: state[0].clone(),
            deterministic,
        };
        self.inference_state_sender
            .as_ref()
            .expect("Should call start() before queue_action().")
            .send(InferenceMessage::ActionMessage(item))
            .expect("Should be able to send message to inference_server");
        Vec::from([action_receiver.recv().unwrap()])
    }

    fn update(&mut self, update: Self::PolicyState) {
        self.inference_state_sender
            .as_ref()
            .expect("Should call start() before update().")
            .send(InferenceMessage::Policy(update))
            .unwrap();
    }

    fn state(&self) -> Self::PolicyState {
        todo!()
    }
}
