use crate::renderer::{tui::NumericMetricsState, MetricsRenderer};
use crate::renderer::{MetricState, TrainingProgress};
use crate::TrainingInterrupter;
use ratatui::{
    crossterm::{
        event::{self, Event, KeyCode},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    },
    prelude::*,
    Terminal,
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
    TextMetricsState,
};

/// The current terminal backend.
pub(crate) type TerminalBackend = CrosstermBackend<Stdout>;
/// The current terminal frame.
pub(crate) type TerminalFrame<'a> = ratatui::Frame<'a>;

#[allow(deprecated)] // `PanicInfo` type is renamed to `PanicHookInfo` in Rust 1.82
type PanicHook = Box<dyn Fn(&std::panic::PanicInfo<'_>) + 'static + Sync + Send>;

const MAX_REFRESH_RATE_MILLIS: u64 = 100;

/// The terminal UI metrics renderer.
pub struct TuiMetricsRenderer {
    terminal: Terminal<TerminalBackend>,
    last_update: std::time::Instant,
    progress: ProgressBarState,
    metrics_numeric: NumericMetricsState,
    metrics_text: TextMetricsState,
    status: StatusState,
    interuptor: TrainingInterrupter,
    popup: PopupState,
    previous_panic_hook: Option<Arc<PanicHook>>,
}

impl MetricsRenderer for TuiMetricsRenderer {
    fn update_train(&mut self, state: MetricState) {
        match state {
            MetricState::Generic(entry) => {
                self.metrics_text.update_train(entry);
            }
            MetricState::Numeric(entry, value) => {
                self.metrics_numeric.push_train(entry.name.clone(), value);
                self.metrics_text.update_train(entry);
            }
        };
    }

    fn update_valid(&mut self, state: MetricState) {
        match state {
            MetricState::Generic(entry) => {
                self.metrics_text.update_valid(entry);
            }
            MetricState::Numeric(entry, value) => {
                self.metrics_numeric.push_valid(entry.name.clone(), value);
                self.metrics_text.update_valid(entry);
            }
        };
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
}

impl TuiMetricsRenderer {
    /// Create a new terminal UI renderer.
    pub fn new(interuptor: TrainingInterrupter, checkpoint: Option<usize>) -> Self {
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
        }
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

                if let Event::Key(key) = event {
                    if let KeyCode::Char('q') = key.code {
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
        }

        Ok(())
    }
}

struct QuitPopupAccept(TrainingInterrupter);
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
            disable_raw_mode().ok();
            execute!(self.terminal.backend_mut(), LeaveAlternateScreen).unwrap();
            self.terminal.show_cursor().ok();

            // Reinstall the previous panic hook
            let _ = take_hook();
            if let Some(previous_panic_hook) =
                Arc::into_inner(self.previous_panic_hook.take().unwrap())
            {
                set_hook(previous_panic_hook);
            }
        }
    }
}
