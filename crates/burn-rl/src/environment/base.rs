pub struct StepResult<S> {
    pub next_state: S,
    pub reward: f64,
    pub done: bool,
    pub truncated: bool,
}

pub trait Environment: Sized + Clone {
    type State;
    type Action;

    const MAX_STEPS: usize;
    const OBS_SPACE: usize;
    const ACTION_SPACE: usize;

    // TODO: New could be removed in favor of letting the user pass a closure to the launch() fn of the ReinforcmentLearning (And remove sized constraint)
    fn new() -> Self;
    fn state(&self) -> Self::State;
    fn step(&mut self, action: Self::Action) -> StepResult<Self::State>;
    fn reset(&mut self);
}

pub trait EnvironmentInit<E: Environment>: Clone {
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
