use crate::metric::{MetricDefinition, MetricId};
use crate::renderer::tui::TuiSplit;
use crate::renderer::{
    EvaluationName, EvaluationProgress, MetricState, MetricsRenderer, MetricsRendererEvaluation,
    ProgressType, TrainingProgress,
};
use crate::renderer::{MetricsRendererTraining, tui::NumericMetricsState};
use crate::{Interrupter, LearnerSummary};
use ratatui::{
    Terminal,
    crossterm::{
        event::{self, Event, KeyCode},
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    },
    prelude::*,
};
use std::collections::HashMap;
use std::panic::{set_hook, take_hook};
use std::sync::{Arc, mpsc};
use std::thread::{JoinHandle, spawn};
use std::{
    error::Error,
    io::{self, Stdout},
    time::{Duration, Instant},
};

use super::{
    Callback, CallbackFn, ControlsView, MetricsView, PopupState, ProgressBarState, StatusState,
    TextMetricsState, TuiGroup, TuiTag,
};

/// The current terminal backend.
pub(crate) type TerminalBackend = CrosstermBackend<Stdout>;
/// The current terminal frame.
pub(crate) type TerminalFrame<'a> = ratatui::Frame<'a>;

type PanicHook = Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + 'static + Sync + Send>;

const MAX_REFRESH_RATE_MILLIS: u64 = 100;

enum TuiRendererEvent {
    MetricRegistration(MetricDefinition),
    MetricsUpdate((TuiSplit, TuiGroup, MetricState)),
    StatusUpdateTrain((TuiSplit, TrainingProgress, Vec<ProgressType>)),
    StatusUpdateTest((EvaluationProgress, Vec<ProgressType>)),
    TrainEnd(Option<LearnerSummary>),
    ManualClose(),
    Close(),
    Persistent(),
}

/// The terminal UI metrics renderer.
pub struct TuiMetricsRendererWrapper {
    sender: mpsc::Sender<TuiRendererEvent>,
    interrupter: Interrupter,
    handle_join: Option<JoinHandle<()>>,
}

impl TuiMetricsRendererWrapper {
    /// Create a new terminal UI renderer.
    pub fn new(interrupter: Interrupter, checkpoint: Option<usize>) -> Self {
        let (sender, receiver) = mpsc::channel();

        let interrupter_clone = interrupter.clone();
        let handle_join = spawn(move || {
            let mut renderer = TuiMetricsRenderer::new(interrupter_clone, checkpoint);

            let tick_rate = Duration::from_millis(MAX_REFRESH_RATE_MILLIS);
            loop {
                match receiver.try_recv() {
                    Ok(event) => renderer.handle_event(event),
                    Err(mpsc::TryRecvError::Empty) => (),
                    Err(mpsc::TryRecvError::Disconnected) => {
                        log::error!("Renderer thread disconnected.");
                        break;
                    }
                }

                // Render
                if renderer.last_update.elapsed() >= tick_rate
                    && let Err(err) = renderer.render()
                {
                    log::error!("Render error: {err}");
                    break;
                }

                if (renderer.manual_close && renderer.interrupter.should_stop()) || renderer.close {
                    break;
                }
            }
        });

        Self {
            sender,
            interrupter,
            handle_join: Some(handle_join),
        }
    }

    fn send_event(&self, event: TuiRendererEvent) {
        if let Err(e) = self.sender.send(event) {
            log::warn!("Failed to send TUI event: {e}");
        }
    }

    /// Set the renderer to persistent mode.
    pub fn persistent(self) -> Self {
        self.send_event(TuiRendererEvent::Persistent());
        self
    }
}

struct TuiMetricsRenderer {
    terminal: Terminal<TerminalBackend>,
    last_update: std::time::Instant,
    progress: ProgressBarState,
    metric_definitions: HashMap<MetricId, MetricDefinition>,
    metrics_numeric: NumericMetricsState,
    metrics_text: TextMetricsState,
    status: StatusState,
    interrupter: Interrupter,
    popup: PopupState,
    previous_panic_hook: Option<Arc<PanicHook>>,
    persistent: bool,
    manual_close: bool,
    close: bool,
    summary: Option<LearnerSummary>,
}

impl MetricsRendererEvaluation for TuiMetricsRendererWrapper {
    fn update_test(&mut self, name: EvaluationName, state: MetricState) {
        self.send_event(TuiRendererEvent::MetricsUpdate((
            TuiSplit::Test,
            TuiGroup::Named(name.name),
            state,
        )));
    }

    fn render_test(&mut self, item: EvaluationProgress, progress_indicators: Vec<ProgressType>) {
        self.send_event(TuiRendererEvent::StatusUpdateTest((
            item,
            progress_indicators,
        )));
    }
}

impl MetricsRenderer for TuiMetricsRendererWrapper {
    fn manual_close(&mut self) {
        self.send_event(TuiRendererEvent::ManualClose());
        let _ = self.handle_join.take().unwrap().join();
    }

    fn register_metric(&mut self, definition: MetricDefinition) {
        self.send_event(TuiRendererEvent::MetricRegistration(definition));
    }
}

impl MetricsRendererTraining for TuiMetricsRendererWrapper {
    fn update_train(&mut self, state: MetricState) {
        self.send_event(TuiRendererEvent::MetricsUpdate((
            TuiSplit::Train,
            TuiGroup::Default,
            state,
        )));
    }

    fn update_valid(&mut self, state: MetricState) {
        self.send_event(TuiRendererEvent::MetricsUpdate((
            TuiSplit::Valid,
            TuiGroup::Default,
            state,
        )));
    }

    fn render_train(&mut self, item: TrainingProgress, progress_indicators: Vec<ProgressType>) {
        self.send_event(TuiRendererEvent::StatusUpdateTrain((
            TuiSplit::Train,
            item,
            progress_indicators,
        )));
    }

    fn render_valid(&mut self, item: TrainingProgress, progress_indicators: Vec<ProgressType>) {
        self.send_event(TuiRendererEvent::StatusUpdateTrain((
            TuiSplit::Valid,
            item,
            progress_indicators,
        )));
    }

    fn on_train_end(&mut self, summary: Option<LearnerSummary>) -> Result<(), Box<dyn Error>> {
        // Reset for following steps.
        self.interrupter.reset();
        // Update the summary
        self.send_event(TuiRendererEvent::TrainEnd(summary));
        Ok(())
    }
}

impl Drop for TuiMetricsRendererWrapper {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            self.send_event(TuiRendererEvent::Close());
            let _ = self.handle_join.take().unwrap().join();
        }
    }
}

impl TuiMetricsRenderer {
    fn update_metric(&mut self, split: TuiSplit, group: TuiGroup, state: MetricState) {
        match state {
            MetricState::Generic(entry) => {
                let name = self
                    .metric_definitions
                    .get(&entry.metric_id)
                    .unwrap()
                    .name
                    .clone()
                    .into();
                self.metrics_text.update(split, group, entry, name);
            }
            MetricState::Numeric(entry, value) => {
                let name: Arc<String> = self
                    .metric_definitions
                    .get(&entry.metric_id)
                    .unwrap()
                    .name
                    .clone()
                    .into();
                self.metrics_numeric
                    .push(TuiTag::new(split, group.clone()), name.clone(), value);
                self.metrics_text.update(split, group, entry, name);
            }
        };
    }

    pub fn new(interrupter: Interrupter, checkpoint: Option<usize>) -> Self {
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen).unwrap();
        enable_raw_mode().unwrap();
        let terminal = Terminal::new(CrosstermBackend::new(stdout)).unwrap();

        // Reset the terminal to raw mode on panic before running the panic handler
        // This prevents that the panic message is not visible for the user.
        let previous_panic_hook = Arc::new(take_hook());
        set_hook(Box::new({
            let previous_panic_hook = previous_panic_hook.clone();
            move |panic_info| {
                let _ = disable_raw_mode();
                let _ = execute!(io::stdout(), LeaveAlternateScreen);
                previous_panic_hook(panic_info);
            }
        }));

        Self {
            terminal,
            last_update: Instant::now(),
            progress: ProgressBarState::new(checkpoint),
            metric_definitions: HashMap::default(),
            metrics_numeric: NumericMetricsState::default(),
            metrics_text: TextMetricsState::default(),
            status: StatusState::default(),
            interrupter,
            popup: PopupState::Empty,
            previous_panic_hook: Some(previous_panic_hook),
            persistent: false,
            manual_close: false,
            close: false,
            summary: None,
        }
    }

    fn handle_event(&mut self, event: TuiRendererEvent) {
        match event {
            TuiRendererEvent::MetricRegistration(definition) => {
                self.metric_definitions
                    .insert(definition.metric_id.clone(), definition);
            }
            TuiRendererEvent::MetricsUpdate((split, group, state)) => {
                self.update_metric(split, group, state);
            }
            TuiRendererEvent::StatusUpdateTrain((split, item, status)) => match split {
                TuiSplit::Train => {
                    self.progress.update_train(&item);
                    self.metrics_numeric.update_progress_train(&item);
                    self.status.update_train(status);
                }
                TuiSplit::Valid => {
                    self.progress.update_valid(&item);
                    self.metrics_numeric.update_progress_valid(&item);
                    self.status.update_valid(status);
                }
                _ => (),
            },
            TuiRendererEvent::StatusUpdateTest((item, status)) => {
                self.progress.update_test(&item);
                self.metrics_numeric.update_progress_test(&item);
                self.status.update_test(status);
            }
            TuiRendererEvent::TrainEnd(learner_summary) => {
                self.interrupter.reset();
                self.summary = learner_summary;
            }
            TuiRendererEvent::ManualClose() => self.manual_close = true,
            TuiRendererEvent::Persistent() => self.persistent = true,
            TuiRendererEvent::Close() => self.close = true,
        }
    }

    fn render(&mut self) -> Result<(), Box<dyn Error>> {
        self.draw()?;
        self.handle_user_input()?;

        self.last_update = Instant::now();

        Ok(())
    }

    fn draw(&mut self) -> Result<(), Box<dyn Error>> {
        self.terminal.draw(|frame| {
            let size = frame.area();

            match self.popup.view() {
                Some(view) => view.render(frame, size),
                None => {
                    let view = MetricsView::new(
                        self.metrics_numeric.view(),
                        self.metrics_text.view(),
                        self.progress.view(),
                        ControlsView,
                        self.status.view(),
                    );

                    view.render(frame, size);
                }
            };
        })?;

        Ok(())
    }

    fn handle_user_input(&mut self) -> Result<(), Box<dyn Error>> {
        while event::poll(Duration::from_secs(0))? {
            let event = event::read()?;
            self.popup.on_event(&event);

            if self.popup.is_empty() {
                self.metrics_numeric.on_event(&event);

                if let Event::Key(key) = event
                    && let KeyCode::Char('q') = key.code
                {
                    self.popup = PopupState::Full(
                        "Quit".to_string(),
                        vec![
                            Callback::new(
                                "Stop the training.",
                                "Stop the training immediately. This will break from the \
                                     training loop, but any remaining code after the loop will be \
                                     executed.",
                                's',
                                QuitPopupAccept(self.interrupter.clone()),
                            ),
                            Callback::new(
                                "Stop the training immediately.",
                                "Kill the program. This will create a panic! which will make \
                                     the current training fails. Any code following the training \
                                     won't be executed.",
                                'k',
                                KillPopupAccept,
                            ),
                            Callback::new(
                                "Cancel",
                                "Cancel the action, continue the training.",
                                'c',
                                PopupCancel,
                            ),
                        ],
                    );
                }
            }
        }

        Ok(())
    }

    fn handle_post_training(&mut self) -> Result<(), Box<dyn Error>> {
        self.popup = PopupState::Full(
            "Training is done".to_string(),
            vec![Callback::new(
                "Training Done",
                "Press 'x' to close this popup.  Press 'q' to exit the application after the \
                popup is closed.",
                'x',
                PopupCancel,
            )],
        );

        self.draw().ok();

        loop {
            if let Ok(true) = event::poll(Duration::from_millis(MAX_REFRESH_RATE_MILLIS)) {
                match event::read() {
                    Ok(event @ Event::Key(key)) => {
                        if self.popup.is_empty() {
                            self.metrics_numeric.on_event(&event);
                            if let KeyCode::Char('q') = key.code {
                                break;
                            }
                        } else {
                            self.popup.on_event(&event);
                        }
                        self.draw().ok();
                    }

                    Ok(Event::Resize(..)) => {
                        self.draw().ok();
                    }
                    Err(err) => {
                        eprintln!("Error reading event: {err}");
                        break;
                    }
                    _ => continue,
                }
            }
        }
        Ok(())
    }

    // Reset the terminal back to raw mode.
    fn reset(&mut self) -> Result<(), Box<dyn Error>> {
        // If previous panic hook has already been re-instated, then the terminal was already reset.
        if self.previous_panic_hook.is_some() {
            if self.persistent
                && let Err(err) = self.handle_post_training()
            {
                eprintln!("Error in post-training handling: {err}");
            }

            disable_raw_mode()?;
            execute!(self.terminal.backend_mut(), LeaveAlternateScreen)?;
            self.terminal.show_cursor()?;

            // Reinstall the previous panic hook
            let _ = take_hook();
            if let Some(previous_panic_hook) =
                Arc::into_inner(self.previous_panic_hook.take().unwrap())
            {
                set_hook(previous_panic_hook);
            }
        }
        Ok(())
    }
}

struct QuitPopupAccept(Interrupter);
struct KillPopupAccept;
struct PopupCancel;

impl CallbackFn for KillPopupAccept {
    fn call(&self) -> bool {
        panic!("Killing training from user input.");
    }
}

impl CallbackFn for QuitPopupAccept {
    fn call(&self) -> bool {
        self.0.stop(Some("Stopping training from user input."));
        true
    }
}

impl CallbackFn for PopupCancel {
    fn call(&self) -> bool {
        true
    }
}

impl Drop for TuiMetricsRenderer {
    fn drop(&mut self) {
        // Reset the terminal back to raw mode. This can be skipped during
        // panicking because the panic hook has already reset the terminal
        if !std::thread::panicking() {
            self.reset().unwrap();

            if let Some(summary) = &self.summary {
                println!("{summary}");
                log::info!("{summary}");
            }
        }
    }
}
