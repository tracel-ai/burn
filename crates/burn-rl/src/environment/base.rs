/// The result of taking a step in an environment.
pub struct StepResult<S> {
    /// The updated state.
    pub next_state: S,
    /// The reward.
    pub reward: f64,
    /// If the environment reached a terminal state.
    pub done: bool,
    /// If the environment reached its max length.
    pub truncated: bool,
}

/// Trait to be implemented for a RL environment.
pub trait Environment {
    /// The type of the state.
    type State;
    /// The type of actions.
    type Action;

    /// The maximum number of step for one episode.
    const MAX_STEPS: usize;
    // TODO: not usize.
    /// The set of all possible states.
    const OBS_SPACE: usize;
    /// The set of all possible actions.
    const ACTION_SPACE: usize;

    /// Returns the current state.
    fn state(&self) -> Self::State;
    /// Take a step in the environment given an action.
    fn step(&mut self, action: Self::Action) -> StepResult<Self::State>;
    /// Reset the environment to an initial state.
    fn reset(&mut self);
}

/// Trait to define how to initialize an environment.
/// By default, any function returning an environment implements it.
pub trait EnvironmentInit<E: Environment>: Clone {
    /// Initialize the environment.
    fn init(&self) -> E;
}

impl<F, E> EnvironmentInit<E> for F
where
    F: Fn() -> E + Clone,
    E: Environment,
{
    fn init(&self) -> E {
        (self)()
    }
}
