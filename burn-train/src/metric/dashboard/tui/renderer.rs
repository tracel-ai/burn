use crate::metric::dashboard::tui::NumericMetricsState;
use crate::metric::dashboard::{DashboardMetricState, DashboardRenderer, TrainingProgress};
use crate::TrainingInterrupter;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{prelude::*, Terminal};
use std::{
    error::Error,
    io::{self, Stdout},
    time::{Duration, Instant},
};

use super::{ControlsView, DashboardView, ProgressState, TextMetricsState};

pub(crate) type TBackend = CrosstermBackend<Stdout>;
pub(crate) type TFrame<'a> = ratatui::Frame<'a, TBackend>;

static MAX_REFRESH_RATE_MILLIS: u64 = 250;

/// The CLI dashboard renderer.
pub struct TuiDashboardRenderer {
    terminal: Terminal<TBackend>,
    last_update: std::time::Instant,
    progress: ProgressState,
    metrics_numeric: NumericMetricsState,
    metrics_text: TextMetricsState,
    interuptor: TrainingInterrupter,
}

impl DashboardRenderer for TuiDashboardRenderer {
    fn update_train(&mut self, state: DashboardMetricState) {
        match state {
            DashboardMetricState::Generic(entry) => {
                self.metrics_text.update_train(entry);
            }
            DashboardMetricState::Numeric(entry, value) => {
                self.metrics_numeric.push_train(entry.name.clone(), value);
                self.metrics_text.update_train(entry);
            }
        };
    }

    fn update_valid(&mut self, state: DashboardMetricState) {
        match state {
            DashboardMetricState::Generic(entry) => {
                self.metrics_text.update_valid(entry);
            }
            DashboardMetricState::Numeric(entry, value) => {
                self.metrics_numeric.push_valid(entry.name.clone(), value);
                self.metrics_text.update_valid(entry);
            }
        };
    }

    fn render_train(&mut self, item: TrainingProgress) {
        self.progress.update_train(item);
        self.render().unwrap();
    }

    fn render_valid(&mut self, item: TrainingProgress) {
        self.progress.update_valid(item);
        self.render().unwrap();
    }
}

impl TuiDashboardRenderer {
    /// Create a new CLI dashboard renderer.
    pub fn new(interuptor: TrainingInterrupter) -> Self {
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen).unwrap();
        enable_raw_mode().unwrap();
        let terminal = Terminal::new(CrosstermBackend::new(stdout)).unwrap();

        Self {
            terminal,
            last_update: Instant::now(),
            progress: ProgressState::default(),
            metrics_numeric: NumericMetricsState::default(),
            metrics_text: TextMetricsState::default(),
            interuptor,
        }
    }

    fn render(&mut self) -> Result<(), Box<dyn Error>> {
        let tick_rate = Duration::from_millis(MAX_REFRESH_RATE_MILLIS);
        if self.last_update.elapsed() < tick_rate {
            return Ok(());
        }

        self.terminal.draw(|mut frame| {
            let size = frame.size();
            let view = DashboardView::new(
                self.metrics_numeric.view(),
                self.metrics_text.view(),
                self.progress.view(),
                ControlsView,
            );

            view.render(&mut frame, size);
        })?;

        while crossterm::event::poll(Duration::from_secs(0))? {
            let event = event::read()?;
            self.metrics_numeric.on_event(&event);

            if let Event::Key(key) = event {
                if let KeyCode::Char('q') = key.code {
                    self.interuptor.stop();
                }
            }
        }

        self.last_update = Instant::now();

        Ok(())
    }
}

impl Drop for TuiDashboardRenderer {
    fn drop(&mut self) {
        disable_raw_mode().unwrap();
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen,).unwrap();
        self.terminal.show_cursor().unwrap();
    }
}
