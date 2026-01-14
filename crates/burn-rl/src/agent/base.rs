use derive_new::new;
use std::{marker::PhantomData, sync::mpsc::Sender, thread::spawn};

use burn_core::{prelude::*, tensor::backend::AutodiffBackend};

use crate::{Environment, Transition, TransitionBuffer};

#[derive(Clone, new)]
pub struct ActionContext<A, C> {
    pub context: C,
    pub action: A,
}

pub trait Policy<B: Backend, S, A> {
    type Context: Clone;

    fn logits(&mut self, state: S) -> Tensor<B, 1>;
    fn action(&mut self, state: S) -> ActionContext<A, Self::Context>;
}

pub trait Agent<B: Backend, E: Environment>: Clone {
    type Policy: Clone + Send;
    type DecisionContext: Clone + Send;

    fn batch_take_action(
        &mut self,
        states: Vec<E::State>,
        deterministic: bool,
    ) -> Vec<ActionContext<E::Action, Self::DecisionContext>>;
    fn take_action(
        &mut self,
        state: E::State,
        deterministic: bool,
    ) -> ActionContext<E::Action, Self::DecisionContext>;
    fn update_policy(&mut self, update: Self::Policy);
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

pub trait LearnerAgent<B: AutodiffBackend, E: Environment>: Agent<B, E> {
    type TrainingInput: From<Transition<B>>;
    type TrainingOutput;

    fn train(
        &mut self,
        input: &TransitionBuffer<Self::TrainingInput>,
    ) -> RLTrainOutput<Self::TrainingOutput, Self::Policy>;
    fn policy(&self) -> Self::Policy;
}

#[derive(Clone)]
struct AutoBatcher<B: Backend, E: Environment, A: Agent<B, E>> {
    autobatch_size: usize,
    inner_agent: A,
    batch: Vec<InferenceItem<E, A::DecisionContext>>,
    _backend: PhantomData<B>,
}

impl<B: Backend, E: Environment + 'static, A: Agent<B, E> + 'static> AutoBatcher<B, E, A> {
    pub fn new(autobatch_size: usize, inner_agent: A) -> Self {
        Self {
            autobatch_size,
            inner_agent,
            batch: vec![],
            _backend: PhantomData,
        }
    }

    pub fn push(&mut self, item: InferenceItem<E, A::DecisionContext>) {
        self.batch.push(item);
        if self.len() >= self.autobatch_size {
            self.flush();
        }
    }

    pub fn len(&self) -> usize {
        self.batch.len()
    }

    pub fn flush(&mut self) {
        let input = self
            .batch
            .iter()
            .map(|m| m.inference_state.clone())
            .collect();
        // Only deterministic if all actions are requested as deterministic.
        let deterministic = self.batch.iter().all(|item| item.deterministic);
        let action_context = self.inner_agent.batch_take_action(input, deterministic);
        for (i, sender) in self.batch.iter().map(|m| m.sender.clone()).enumerate() {
            sender.send(action_context[i].clone()).unwrap();
        }
        self.batch.clear();
    }

    pub fn update_policy(&mut self, policy_update: A::Policy) {
        if self.len() > 0 {
            self.flush();
        }
        self.inner_agent.update_policy(policy_update);
    }
}

enum InferenceMessage<B: Backend, E: Environment, A: Agent<B, E>> {
    Item(InferenceItem<E, A::DecisionContext>),
    Policy(A::Policy),
}

#[derive(Clone)]
struct InferenceItem<E: Environment, C> {
    sender: Sender<ActionContext<E::Action, C>>,
    inference_state: E::State,
    deterministic: bool,
}

#[derive(Clone)]
pub struct AsyncAgent<B: Backend, E: Environment, A: Agent<B, E>> {
    inference_state_sender: Option<Sender<InferenceMessage<B, E, A>>>,
}

impl<B: Backend, E: Environment + 'static, A: Agent<B, E> + Send + 'static> AsyncAgent<B, E, A> {
    pub fn new(autobatch_size: usize, inner_agent: A) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();
        let mut autobatcher = AutoBatcher::new(autobatch_size, inner_agent.clone());
        spawn(move || {
            loop {
                match receiver
                    .recv()
                    .expect("Should be able to receive inference messages.")
                {
                    InferenceMessage::Item(item) => autobatcher.push(item),
                    InferenceMessage::Policy(update) => autobatcher.update_policy(update),
                }
            }
        });

        Self {
            inference_state_sender: Some(sender),
        }
    }
}

impl<B: Backend, E: Environment + 'static, A: Agent<B, E>> Agent<B, E> for AsyncAgent<B, E, A> {
    type Policy = A::Policy;
    type DecisionContext = A::DecisionContext;

    fn batch_take_action(
        &mut self,
        states: Vec<E::State>,
        deterministic: bool,
    ) -> Vec<ActionContext<E::Action, Self::DecisionContext>> {
        let (action_sender, action_receiver) = std::sync::mpsc::channel();
        // Assume that only one state is passed.
        let item = InferenceItem {
            sender: action_sender,
            inference_state: states[0].clone(),
            deterministic,
        };
        self.inference_state_sender
            .as_ref()
            .expect("Should call start() before queue_action().")
            .send(InferenceMessage::Item(item))
            .expect("Should be able to send message to inference_server");
        Vec::from([action_receiver.recv().unwrap()])
    }

    fn update_policy(&mut self, update: Self::Policy) {
        self.inference_state_sender
            .as_ref()
            .expect("Should call start() before update().")
            .send(InferenceMessage::Policy(update))
            .unwrap();
    }

    fn take_action(
        &mut self,
        state: E::State,
        deterministic: bool,
    ) -> ActionContext<E::Action, A::DecisionContext> {
        self.batch_take_action(vec![state], deterministic)[0].clone()
    }
}
