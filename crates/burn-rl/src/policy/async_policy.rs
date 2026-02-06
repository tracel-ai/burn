use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc::{self, Sender},
    },
    thread::spawn,
};

use burn_core::prelude::Backend;

use crate::{ActionContext, Batchable, Policy, PolicyState};

#[derive(Clone)]
struct PolicyInferenceServer<B: Backend, P: Policy<B>> {
    // `num_agents` used to make sure autobatching doesn't block the agents if they are less than the autobatch size.
    num_agents: Arc<AtomicUsize>,
    max_autobatch_size: usize,
    inner_policy: P,
    batch_action: Vec<ActionItem<P::Observation, P::Action, P::ActionContext>>,
    batch_logits: Vec<ForwardItem<P::Observation, P::ActionDistribution>>,
}

impl<B, P> PolicyInferenceServer<B, P>
where
    B: Backend,
    P: Policy<B>,
    P::Observation: Clone + Batchable,
    P::ActionDistribution: Clone + Batchable,
    P::Action: Clone + Batchable,
    P::ActionContext: Clone,
{
    pub fn new(max_autobatch_size: usize, inner_policy: P) -> Self {
        Self {
            num_agents: Arc::new(AtomicUsize::new(0)),
            max_autobatch_size,
            inner_policy,
            batch_action: vec![],
            batch_logits: vec![],
        }
    }

    pub fn push_action(&mut self, item: ActionItem<P::Observation, P::Action, P::ActionContext>) {
        self.batch_action.push(item);
        if self.len_actions()
            >= self
                .num_agents
                .load(Ordering::Relaxed)
                .min(self.max_autobatch_size)
        {
            self.flush_actions();
        }
    }

    pub fn push_logits(&mut self, item: ForwardItem<P::Observation, P::ActionDistribution>) {
        self.batch_logits.push(item);
        if self.len_logits()
            >= self
                .num_agents
                .load(Ordering::Relaxed)
                .min(self.max_autobatch_size)
        {
            self.flush_logits();
        }
    }

    pub fn len_actions(&self) -> usize {
        self.batch_action.len()
    }

    pub fn len_logits(&self) -> usize {
        self.batch_logits.len()
    }

    pub fn flush_actions(&mut self) {
        if self.len_actions() == 0 {
            return;
        }
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
        if self.len_logits() == 0 {
            return;
        }
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
        self.batch_logits.clear();
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

    pub fn increment_agents(&mut self, num: usize) {
        self.num_agents.fetch_add(num, Ordering::Relaxed);
    }

    pub fn decrement_agents(&mut self, num: usize) {
        self.num_agents.fetch_sub(num, Ordering::Relaxed);
        if self.len_actions()
            >= self
                .num_agents
                .load(Ordering::Relaxed)
                .min(self.max_autobatch_size)
        {
            self.flush_actions();
        }
        if self.len_logits()
            >= self
                .num_agents
                .load(Ordering::Relaxed)
                .min(self.max_autobatch_size)
        {
            self.flush_logits();
        }
    }
}

enum InferenceMessage<B: Backend, P: Policy<B>> {
    ActionMessage(ActionItem<P::Observation, P::Action, P::ActionContext>),
    ForwardMessage(ForwardItem<P::Observation, P::ActionDistribution>),
    PolicyUpdate(P::PolicyState),
    PolicyRequest(Sender<P::PolicyState>),
    IncrementAgents(usize),
    DecrementAgents(usize),
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

/// An asynchronous policy using an inference server with autobatching.
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
    /// * `autobatch_size` - Number of observations to accumulate before running a pass of inference.
    /// * `inner_policy` - The policy used to take actions.
    pub fn new(autobatch_size: usize, inner_policy: P) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();
        let mut autobatcher = PolicyInferenceServer::new(autobatch_size, inner_policy.clone());
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
                        InferenceMessage::IncrementAgents(num) => autobatcher.increment_agents(num),
                        InferenceMessage::DecrementAgents(num) => autobatcher.decrement_agents(num),
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

    /// Increment the number of agents using the inference server.
    pub fn increment_agents(&self, num: usize) {
        self.inference_state_sender
            .send(InferenceMessage::IncrementAgents(num))
            .expect("Can send message to autobatcher.")
    }

    /// Decrement the number of agents using the inference server.
    pub fn decrement_agents(&self, num: usize) {
        self.inference_state_sender
            .send(InferenceMessage::DecrementAgents(num))
            .expect("Can send message to autobatcher.")
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

    fn load_record(self, _record: <Self::PolicyState as PolicyState<B>>::Record) -> Self {
        // Not needed for now
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use std::thread::JoinHandle;
    use std::time::Duration;

    use crate::TestBackend;
    use crate::tests::{MockAction, MockObservation, MockPolicy};

    use super::*;

    #[test]
    fn test_multiple_actions_before_flush() {
        fn launch_thread(
            policy: &AsyncPolicy<TestBackend, MockPolicy>,
            handles: &mut Vec<JoinHandle<()>>,
        ) {
            let mut thread_policy = policy.clone();
            let handle = spawn(move || {
                thread_policy.action(MockObservation(vec![0.]), false);
            });
            handles.push(handle);
        }

        let policy = AsyncPolicy::new(8, MockPolicy::new());
        policy.increment_agents(1000);

        let mut handles = vec![];
        launch_thread(&policy, &mut handles);
        std::thread::sleep(Duration::from_millis(10));
        assert!(!handles[0].is_finished());

        for _ in 0..6 {
            launch_thread(&policy, &mut handles);
        }
        std::thread::sleep(Duration::from_millis(10));
        for i in 0..7 {
            assert!(!handles[i].is_finished());
        }

        launch_thread(&policy, &mut handles);
        std::thread::sleep(Duration::from_millis(10));
        for i in 0..8 {
            assert!(handles[i].is_finished());
        }

        let mut handles = vec![];
        launch_thread(&policy, &mut handles);
        std::thread::sleep(Duration::from_millis(10));
        assert!(!handles[0].is_finished());
    }

    #[test]
    fn test_multiple_forward_before_flush() {
        fn launch_thread(
            policy: &AsyncPolicy<TestBackend, MockPolicy>,
            handles: &mut Vec<JoinHandle<()>>,
        ) {
            let mut thread_policy = policy.clone();
            let handle = spawn(move || {
                thread_policy.forward(MockObservation(vec![0.]));
            });
            handles.push(handle);
        }

        let policy = AsyncPolicy::new(8, MockPolicy::new());
        policy.increment_agents(1000);

        let mut handles = vec![];
        launch_thread(&policy, &mut handles);
        std::thread::sleep(Duration::from_millis(10));
        assert!(!handles[0].is_finished());

        for _ in 0..6 {
            launch_thread(&policy, &mut handles);
        }
        std::thread::sleep(Duration::from_millis(10));
        for i in 0..7 {
            assert!(!handles[i].is_finished());
        }

        launch_thread(&policy, &mut handles);
        std::thread::sleep(Duration::from_millis(10));
        for i in 0..8 {
            assert!(handles[i].is_finished());
        }

        let mut handles = vec![];
        launch_thread(&policy, &mut handles);
        std::thread::sleep(Duration::from_millis(10));
        assert!(!handles[0].is_finished());
    }

    #[test]
    fn test_async_policy_deterministic_behaviour() {
        fn launch_thread(
            policy: &AsyncPolicy<TestBackend, MockPolicy>,
            handles: &mut Vec<JoinHandle<MockAction>>,
            deterministic: bool,
        ) {
            let mut thread_policy = policy.clone();
            let handle = spawn(move || {
                let (action, _) = thread_policy.action(MockObservation(vec![0.]), deterministic);
                action
            });
            handles.push(handle);
        }

        let policy = AsyncPolicy::new(2, MockPolicy::new());
        policy.increment_agents(1000);

        let mut handles = vec![];
        launch_thread(&policy, &mut handles, true);
        launch_thread(&policy, &mut handles, false);
        for _ in 0..2 {
            let action = handles.pop().unwrap().join().unwrap();
            assert_eq!(action.0, vec![0]);
        }

        let mut handles = vec![];
        launch_thread(&policy, &mut handles, true);
        launch_thread(&policy, &mut handles, true);
        for _ in 0..2 {
            let action = handles.pop().unwrap().join().unwrap();
            assert_eq!(action.0, vec![1]);
        }
    }

    #[test]
    fn flush_when_running_agents_smaller_than_autobatch_size() {
        fn launch_thread(
            policy: &AsyncPolicy<TestBackend, MockPolicy>,
            handles: &mut Vec<JoinHandle<()>>,
        ) {
            let mut thread_policy = policy.clone();
            let handle = spawn(move || {
                thread_policy.action(MockObservation(vec![0.]), false);
            });
            handles.push(handle);
        }

        let policy = AsyncPolicy::new(8, MockPolicy::new());
        policy.increment_agents(3);

        let mut handles = vec![];
        launch_thread(&policy, &mut handles);
        launch_thread(&policy, &mut handles);
        std::thread::sleep(Duration::from_millis(10));
        assert!(!handles[0].is_finished());
        assert!(!handles[1].is_finished());

        launch_thread(&policy, &mut handles);
        std::thread::sleep(Duration::from_millis(10));
        for i in 0..3 {
            assert!(handles[i].is_finished());
        }

        let mut handles = vec![];
        launch_thread(&policy, &mut handles);
        launch_thread(&policy, &mut handles);
        std::thread::sleep(Duration::from_millis(10));
        assert!(!handles[0].is_finished());
        assert!(!handles[1].is_finished());

        policy.decrement_agents(1);
        std::thread::sleep(Duration::from_millis(10));
        assert!(handles[0].is_finished());
        assert!(handles[1].is_finished());
    }
}
