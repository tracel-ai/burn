use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc::{self, Sender},
    },
    thread::spawn,
};

use burn_core::tensor::Device;

use crate::{ActionContext, Batchable, Policy, PolicyState};

#[derive(Clone)]
struct PolicyInferenceServer<P: Policy> {
    // `num_agents` used to make sure autobatching doesn't block the agents if they are less than the autobatch size.
    num_agents: Arc<AtomicUsize>,
    max_autobatch_size: usize,
    inner_policy: P,
    batch_action: Vec<ActionItem<P::Observation, P::Action, P::ActionContext>>,
    batch_logits: Vec<ForwardItem<P::Observation, P::ActionDistribution>>,
}

impl<P> PolicyInferenceServer<P>
where
    P: Policy,
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
        let batch_action = core::mem::take(&mut self.batch_action);
        let first_deterministic = batch_action[0].deterministic;

        // Keep each request's deterministic mode while retaining autobatching within each group.
        // Process the first queued mode first to avoid unnecessarily reordering policy calls.
        for deterministic in [first_deterministic, !first_deterministic] {
            let (indices, input): (Vec<_>, Vec<_>) = batch_action
                .iter()
                .enumerate()
                .filter(|(_, item)| item.deterministic == deterministic)
                .map(|(index, item)| (index, item.inference_state.clone()))
                .unzip();

            if input.is_empty() {
                continue;
            }

            let (actions, context) = self
                .inner_policy
                .action(P::Observation::batch(input), deterministic);
            let actions: Vec<_> = actions.unbatch();

            for (group_index, item_index) in indices.into_iter().enumerate() {
                batch_action[item_index]
                    .sender
                    .send(ActionContext {
                        context: vec![context[group_index].clone()],
                        action: actions[group_index].clone(),
                    })
                    .expect("Autobatcher should be able to send resulting actions.");
            }
        }
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

    pub fn policy_to_device(&mut self, device: &Device) {
        self.inner_policy = self.inner_policy.clone().to_device(device);
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

enum InferenceMessage<P: Policy> {
    ActionMessage(ActionItem<P::Observation, P::Action, P::ActionContext>),
    ForwardMessage(ForwardItem<P::Observation, P::ActionDistribution>),
    PolicyUpdate(P::PolicyState),
    ToDevice(Device),
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
pub struct AsyncPolicy<P: Policy> {
    inference_state_sender: Sender<InferenceMessage<P>>,
}

impl<P> AsyncPolicy<P>
where
    P: Policy + Clone + Send + 'static,
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
                        InferenceMessage::ToDevice(device) => autobatcher.policy_to_device(&device),
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

impl<P> Policy for AsyncPolicy<P>
where
    P: Policy + Send + 'static,
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

    fn to_device(self, device: &Device) -> Self {
        self.inference_state_sender
            .send(InferenceMessage::ToDevice(device.clone()))
            .expect("AsyncPolicy should be able to send policy state.");
        self
    }

    fn load_record(self, _record: <Self::PolicyState as PolicyState>::Record) -> Self {
        unimplemented!(
            "Not implemented yet. Please load the record on the inner policy before creating an async policy."
        )
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use std::sync::Mutex;
    use std::thread::JoinHandle;
    use std::time::Duration;

    use crate::tests::{
        MockAction, MockActionDistribution, MockObservation, MockPolicy, MockPolicyState,
    };

    use super::*;

    type PolicyCalls = Arc<Mutex<Vec<(bool, Vec<f32>)>>>;

    #[derive(Clone)]
    struct TrackingContext(i32);

    #[derive(Clone)]
    struct TrackingPolicy {
        calls: PolicyCalls,
        inner: MockPolicy,
    }

    impl TrackingPolicy {
        fn new(calls: PolicyCalls) -> Self {
            Self {
                calls,
                inner: MockPolicy::new(),
            }
        }
    }

    impl Policy for TrackingPolicy {
        type Observation = MockObservation;
        type ActionDistribution = MockActionDistribution;
        type Action = MockAction;
        type ActionContext = TrackingContext;
        type PolicyState = MockPolicyState;

        fn forward(&mut self, obs: Self::Observation) -> Self::ActionDistribution {
            self.inner.forward(obs)
        }

        fn action(
            &mut self,
            obs: Self::Observation,
            deterministic: bool,
        ) -> (Self::Action, Vec<Self::ActionContext>) {
            self.calls
                .lock()
                .unwrap()
                .push((deterministic, obs.0.clone()));

            let offset = if deterministic { 100 } else { 0 };
            let contexts = obs
                .0
                .iter()
                .map(|value| TrackingContext(*value as i32))
                .collect();
            let actions = obs
                .0
                .into_iter()
                .map(|value| MockAction(vec![value as i32 + offset]))
                .collect();

            (MockAction::batch(actions), contexts)
        }

        fn update(&mut self, update: Self::PolicyState) {
            self.inner.update(update);
        }

        fn state(&self) -> Self::PolicyState {
            self.inner.state()
        }

        fn to_device(mut self, device: &Device) -> Self {
            self.inner = self.inner.to_device(device);
            self
        }

        fn load_record(mut self, record: <Self::PolicyState as PolicyState>::Record) -> Self {
            self.inner = self.inner.load_record(record);
            self
        }
    }

    #[test]
    fn test_multiple_actions_before_flush() {
        fn launch_thread(policy: &AsyncPolicy<MockPolicy>, handles: &mut Vec<JoinHandle<()>>) {
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
        fn launch_thread(policy: &AsyncPolicy<MockPolicy>, handles: &mut Vec<JoinHandle<()>>) {
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
            policy: &AsyncPolicy<MockPolicy>,
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
        let deterministic_action = handles.remove(0).join().unwrap();
        let stochastic_action = handles.remove(0).join().unwrap();
        assert_eq!(deterministic_action.0, vec![1]);
        assert_eq!(stochastic_action.0, vec![0]);

        for deterministic in [false, true] {
            let mut handles = vec![];
            launch_thread(&policy, &mut handles, deterministic);
            launch_thread(&policy, &mut handles, deterministic);
            for _ in 0..2 {
                let action = handles.pop().unwrap().join().unwrap();
                assert_eq!(action.0, vec![i32::from(deterministic)]);
            }
        }
    }

    #[test]
    fn test_action_batch_preserves_request_mapping_and_group_order() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut server = PolicyInferenceServer::new(4, TrackingPolicy::new(calls.clone()));
        server.increment_agents(1000);

        let requests = [(10., true), (20., false), (30., true), (40., false)];
        let mut receivers = Vec::new();
        for (value, deterministic) in requests {
            let (sender, receiver) = mpsc::channel();
            receivers.push(receiver);
            server.push_action(ActionItem {
                sender,
                inference_state: MockObservation(vec![value]),
                deterministic,
            });
        }

        let results: Vec<_> = receivers
            .into_iter()
            .map(|receiver| {
                let result = receiver.recv().unwrap();
                (result.action.0[0], result.context[0].0)
            })
            .collect();
        assert_eq!(results, vec![(110, 10), (20, 20), (130, 30), (40, 40)]);
        assert_eq!(
            *calls.lock().unwrap(),
            vec![(true, vec![10., 30.]), (false, vec![20., 40.])]
        );
    }

    #[test]
    fn flush_when_running_agents_smaller_than_autobatch_size() {
        fn launch_thread(policy: &AsyncPolicy<MockPolicy>, handles: &mut Vec<JoinHandle<()>>) {
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
