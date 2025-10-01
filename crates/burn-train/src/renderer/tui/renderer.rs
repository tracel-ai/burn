use crate::Interrupter;
use crate::renderer::tui::TuiSplit;
use crate::renderer::{
    EvaluationName, EvaluationProgress, MetricState, MetricsRenderer, MetricsRendererEvaluation,
    TrainingProgress,
};
use crate::renderer::{MetricsRendererTraining, tui::NumericMetricsState};
use ratatui::{
    Terminal,
    crossterm::{
        event::{self, Event, KeyCode},
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    },
    prelude::*,
};
use std::panic::{set_hook, take_hook};
use std::sync::Arc;
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

/// The terminal UI metrics renderer.
pub struct TuiMetricsRenderer {
    terminal: Terminal<TerminalBackend>,
    last_update: std::time::Instant,
    progress: ProgressBarState,
    metrics_numeric: NumericMetricsState,
    metrics_text: TextMetricsState,
    status: StatusState,
    interuptor: Interrupter,
    popup: PopupState,
    previous_panic_hook: Option<Arc<PanicHook>>,
    persistent: bool,
}

impl MetricsRendererEvaluation for TuiMetricsRenderer {
    fn update_test(&mut self, name: EvaluationName, state: MetricState) {
        self.update_metric(TuiSplit::Test, TuiGroup::Named(name.name), state);
    }

    fn render_test(&mut self, item: EvaluationProgress) {
        self.progress.update_test(&item);
        self.metrics_numeric.update_progress_test(&item);
        self.status.update_test(item);
        self.render().unwrap();
    }
}

impl MetricsRenderer for TuiMetricsRenderer {
    fn manual_close(&mut self) {
        loop {
            self.render().unwrap();
            if self.interuptor.should_stop() {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

impl MetricsRendererTraining for TuiMetricsRenderer {
    fn update_train(&mut self, state: MetricState) {
        self.update_metric(TuiSplit::Train, TuiGroup::Default, state);
    }

    fn update_valid(&mut self, state: MetricState) {
        self.update_metric(TuiSplit::Valid, TuiGroup::Default, state);
    }

    fn render_train(&mut self, item: TrainingProgress) {
        self.progress.update_train(&item);
        self.metrics_numeric.update_progress_train(&item);
        self.status.update_train(item);
        self.render().unwrap();
    }

    fn render_valid(&mut self, item: TrainingProgress) {
        self.progress.update_valid(&item);
        self.metrics_numeric.update_progress_valid(&item);
        self.status.update_valid(item);
        self.render().unwrap();
    }

    fn on_train_end(&mut self) -> Result<(), Box<dyn Error>> {
        // Reset for following steps.
        self.interuptor.reset();
        Ok(())
    }
}

impl TuiMetricsRenderer {
    fn update_metric(&mut self, split: TuiSplit, group: TuiGroup, state: MetricState) {
        match state {
            MetricState::Generic(entry) => {
                self.metrics_text.update(split, group, entry);
            }
            MetricState::Numeric(entry, value) => {
                self.metrics_numeric.push(
                    TuiTag::new(split, group.clone()),
                    entry.name.clone(),
                    value,
                );
                self.metrics_text.update(split, group, entry);
            }
        };
    }

    /// Create a new terminal UI renderer.
    pub fn new(interuptor: Interrupter, checkpoint: Option<usize>) -> Self {
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
            metrics_numeric: NumericMetricsState::default(),
            metrics_text: TextMetricsState::default(),
            status: StatusState::default(),
            interuptor,
            popup: PopupState::Empty,
            previous_panic_hook: Some(previous_panic_hook),
            persistent: false,
        }
    }

    /// Set the renderer to persistent mode.
    pub fn persistent(mut self) -> Self {
        self.persistent = true;
        self
    }

    fn render(&mut self) -> Result<(), Box<dyn Error>> {
        let tick_rate = Duration::from_millis(MAX_REFRESH_RATE_MILLIS);
        if self.last_update.elapsed() < tick_rate {
            return Ok(());
        }

        self.draw()?;
        self.handle_events()?;

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

    fn handle_events(&mut self) -> Result<(), Box<dyn Error>> {
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
                                QuitPopupAccept(self.interuptor.clone()),
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
        self.0.stop();
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
        }
    }
}
