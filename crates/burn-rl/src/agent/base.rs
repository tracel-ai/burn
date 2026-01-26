use derive_new::new;
use std::{marker::PhantomData, sync::mpsc::Sender, thread::spawn};

use burn_core::{prelude::*, record::Record, tensor::backend::AutodiffBackend};

use crate::{Transition, TransitionBuffer};

#[derive(Clone, new)]
pub struct ActionContext<A, C> {
    pub context: C,
    pub action: A,
}

pub trait PolicyState<B: Backend> {
    type Record: Record<B>;

    fn into_record(&self) -> Self::Record;
    fn load_record(&self, record: Self::Record) -> Self;
}

pub trait Policy<B: Backend>: Clone {
    type Input: Clone + Send;
    // TODO: naming?
    type Logits: Clone + Send;
    // TODO: This VS ComponentsTypes trait.
    type Action: Clone + Send;

    type ActionContext: Clone;
    type PolicyState: Clone + Send + PolicyState<B>;

    fn logits(&mut self, states: Self::Input) -> Self::Logits;
    fn action(
        &mut self,
        states: Self::Input,
        deterministic: bool,
    ) -> (Self::Action, Vec<Self::ActionContext>);

    // TODO: A littel weird but idk.
    fn batch(&self, inputs: Vec<&Self::Input>) -> Self::Input;
    fn unbatch(&self, inputs: Self::Action) -> Vec<Self::Action>;

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

pub trait LearnerAgent<B: AutodiffBackend> {
    type TrainingInput: From<
        Transition<
            B,
            <Self::InnerPolicy as Policy<B>>::Input,
            <Self::InnerPolicy as Policy<B>>::Action,
        >,
    >;
    type TrainingOutput;
    type InnerPolicy: Policy<B>;
    type Record: Record<B>;

    fn train(
        &mut self,
        input: &TransitionBuffer<Self::TrainingInput>,
    ) -> RLTrainOutput<Self::TrainingOutput, <Self::InnerPolicy as Policy<B>>::PolicyState>;
    fn policy(&self) -> Self::InnerPolicy;
    fn update_policy(&mut self, update: Self::InnerPolicy);

    fn into_record(&self) -> Self::Record;
    fn load_record(self, record: Self::Record) -> Self;
}

#[derive(Clone)]
struct AutoBatcher<B: Backend, P: Policy<B>> {
    autobatch_size: usize,
    inner_policy: P,
    batch_action: Vec<ActionItem<P::Input, P::Action, P::ActionContext>>,
    batch_logits: Vec<LogitsItem<P::Input, P::Logits>>,
    _backend: PhantomData<B>,
}

impl<B, P> AutoBatcher<B, P>
where
    B: Backend,
    P: Policy<B>,
    P::Action: Clone,
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

    pub fn push_action(&mut self, item: ActionItem<P::Input, P::Action, P::ActionContext>) {
        self.batch_action.push(item);
        if self.len_actions() >= self.autobatch_size {
            self.flush_actions();
        }
    }

    pub fn push_logits(&mut self, item: LogitsItem<P::Input, P::Logits>) {
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
        let (actions, context) = self
            .inner_policy
            .action(self.inner_policy.batch(input), deterministic);
        let actions = self.inner_policy.unbatch(actions);

        for (i, item) in self.batch_action.iter().enumerate() {
            item.sender
                .send(ActionContext {
                    context: vec![context[i].clone()],
                    action: actions[i].clone(),
                })
                .unwrap();
        }
        self.batch_action.clear();
    }

    // pub fn flush_logits(&mut self) {
    //     let input = self
    //         .batch_logits
    //         .iter()
    //         .map(|m| &m.inference_state)
    //         .collect();
    //     let output = self.inner_policy.logits(self.inner_policy.batch(input));
    //     for (i, item) in self.batch_logits.iter().enumerate() {
    //         item.sender.send(output.clone().slice(s![i, ..])).unwrap();
    //     }
    //     self.batch_action.clear();
    // }

    pub fn update_policy(&mut self, policy_update: P::PolicyState) {
        if self.len_actions() > 0 {
            self.flush_actions();
        }
        // if self.len_logits() > 0 {
        //     self.flush_logits();
        // }
        self.inner_policy.update(policy_update);
    }
}

enum InferenceMessage<B: Backend, P: Policy<B>> {
    ActionMessage(ActionItem<P::Input, P::Action, P::ActionContext>),
    LogitsMessage(LogitsItem<P::Input, P::Logits>),
    Policy(P::PolicyState),
}

#[derive(Clone)]
struct ActionItem<S, A, C> {
    sender: Sender<ActionContext<A, Vec<C>>>,
    inference_state: S,
    deterministic: bool,
}

#[derive(Clone)]
struct LogitsItem<S, L> {
    sender: Sender<L>,
    inference_state: S,
}

#[derive(Clone)]
pub struct AsyncPolicy<B: Backend, P: Policy<B>> {
    inference_state_sender: Option<Sender<InferenceMessage<B, P>>>,
}

impl<B, P> AsyncPolicy<B, P>
where
    B: Backend,
    P: Policy<B> + Clone + Send + 'static,
    P::ActionContext: Send,
    P::PolicyState: Send,
    P::Input: Send,
    P::Logits: Send,
    P::Action: Clone + Send,
{
    pub fn new(autobatch_size: usize, inner_policy: P) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();
        let mut autobatcher = AutoBatcher::new(autobatch_size, inner_policy.clone());
        spawn(move || {
            loop {
                match receiver.recv() {
                    Ok(msg) => match msg {
                        InferenceMessage::ActionMessage(item) => autobatcher.push_action(item),
                        InferenceMessage::LogitsMessage(item) => autobatcher.push_logits(item),
                        InferenceMessage::Policy(update) => autobatcher.update_policy(update),
                    },
                    Err(err) => {
                        log::error!("Error in AsyncPolicy : {}", err);
                        break;
                    }
                }
            }
        });

        Self {
            inference_state_sender: Some(sender),
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

    type Input = P::Input;
    type Logits = P::Logits;
    type Action = P::Action;

    fn logits(&mut self, states: Self::Input) -> Self::Logits {
        let (action_sender, action_receiver) = std::sync::mpsc::channel();
        // TODO: Don't Assume that only one state is passed.
        let item = LogitsItem {
            sender: action_sender,
            inference_state: states,
        };
        self.inference_state_sender
            .as_ref()
            .expect("Should call start() before queue_action().")
            .send(InferenceMessage::LogitsMessage(item))
            .expect("Should be able to send message to inference_server");
        action_receiver.recv().unwrap()
    }

    fn action(
        &mut self,
        states: Self::Input,
        deterministic: bool,
    ) -> (Self::Action, Vec<Self::ActionContext>) {
        let (action_sender, action_receiver) = std::sync::mpsc::channel();
        // TODO: Don't Assume that only one state is passed.
        let item = ActionItem {
            sender: action_sender,
            inference_state: states,
            deterministic,
        };
        self.inference_state_sender
            .as_ref()
            .expect("Should call start() before queue_action().")
            .send(InferenceMessage::ActionMessage(item))
            .expect("Should be able to send message to inference_server");
        let action = action_receiver.recv().unwrap();
        (action.action, action.context)
    }

    fn update(&mut self, update: Self::PolicyState) {
        self.inference_state_sender
            .as_ref()
            .expect("Should call start() before update().")
            .send(InferenceMessage::Policy(update))
            .unwrap();
    }

    // TODO: all this.
    fn state(&self) -> Self::PolicyState {
        todo!()
    }

    fn batch(&self, _inputs: Vec<&Self::Input>) -> Self::Input {
        todo!()
    }

    fn unbatch(&self, _inputs: Self::Action) -> Vec<Self::Action> {
        todo!()
    }
}
