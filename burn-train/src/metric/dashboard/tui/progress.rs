use super::TFrame;
use crate::metric::dashboard::TrainingProgress;
use ratatui::{
    prelude::{Alignment, Rect},
    style::{Color, Style},
    widgets::{Block, Borders, Gauge},
};

pub(crate) struct ProgressState {
    progress: TrainingProgress,
    mode: Mode,
}

enum Mode {
    Valid,
    Train,
}

impl Default for ProgressState {
    fn default() -> Self {
        Self {
            progress: TrainingProgress::none(),
            mode: Mode::Train,
        }
    }
}

impl ProgressState {
    pub(crate) fn update_train(&mut self, progress: TrainingProgress) {
        self.progress = progress;
        self.mode = Mode::Train;
    }
    pub(crate) fn update_valid(&mut self, progress: TrainingProgress) {
        self.progress = progress;
        self.mode = Mode::Valid;
    }

    pub(crate) fn view<'a>(&'a self) -> ProgressView<'a> {
        let epoch = format!(
            "Epoch {}/{}",
            self.progress.epoch, self.progress.epoch_total
        );
        let items = format!(
            "Items {}/{}",
            self.progress.progress.items_processed, self.progress.progress.items_total
        );
        let iteration = format!("Iteration {}", self.progress.iteration,);
        let progress = format!("{epoch} - {items} - {iteration}");
        let title = match self.mode {
            Mode::Valid => format!(" Validation | {} ", progress),
            Mode::Train => format!(" Training | {} ", progress),
        };

        ProgressView::new(&self.progress, title)
    }
}

#[derive(new)]
pub(crate) struct ProgressView<'a> {
    progress: &'a TrainingProgress,
    title: String,
}

impl<'a> ProgressView<'a> {
    pub(crate) fn render<'b>(self, frame: &mut TFrame<'b>, size: Rect) {
        let percent_items = (self.progress.progress.items_processed as f32
            / self.progress.progress.items_total as f32)
            * 100.;

        let iteration = Gauge::default()
            .block(
                Block::default()
                    .title(self.title)
                    .title_alignment(Alignment::Center)
                    .borders(Borders::ALL),
            )
            .gauge_style(Style::default().fg(Color::Yellow))
            .percent(percent_items as u16);

        frame.render_widget(iteration, size)
    }
}
