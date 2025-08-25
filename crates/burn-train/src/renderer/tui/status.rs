use super::TerminalFrame;
use crate::renderer::{EvaluationProgress, TrainingProgress};
use ratatui::{
    prelude::{Alignment, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

/// Show the training status with various information.
pub(crate) struct StatusState {
    progress: TrainingProgress,
    mode: Mode,
}

enum Mode {
    Valid,
    Train,
    Evaluation,
}

impl Default for StatusState {
    fn default() -> Self {
        Self {
            progress: TrainingProgress::none(),
            mode: Mode::Train,
        }
    }
}

impl StatusState {
    /// Update the training information.
    pub(crate) fn update_train(&mut self, progress: TrainingProgress) {
        self.progress = progress;
        self.mode = Mode::Train;
    }
    /// Update the validation information.
    pub(crate) fn update_valid(&mut self, progress: TrainingProgress) {
        self.progress = progress;
        self.mode = Mode::Valid;
    }
    /// Update the testing information.
    pub(crate) fn update_test(&mut self, _progress: EvaluationProgress) {
        // TODO: Use the progress here.
        // self.progress = progress;
        self.mode = Mode::Evaluation;
    }
    /// Create a view.
    pub(crate) fn view(&self) -> StatusView {
        StatusView::new(&self.progress, &self.mode)
    }
}

pub(crate) struct StatusView {
    lines: Vec<Vec<Span<'static>>>,
}

impl StatusView {
    fn new(progress: &TrainingProgress, mode: &Mode) -> Self {
        let title = |title: &str| Span::from(format!(" {title} ")).bold().yellow();
        let value = |value: String| Span::from(value).italic();
        let mode = match mode {
            Mode::Valid => "Validating",
            Mode::Train => "Training",
            Mode::Evaluation => "Evaluation",
        };

        Self {
            lines: vec![
                vec![title("Mode      :"), value(mode.to_string())],
                vec![
                    title("Epoch     :"),
                    value(format!("{}/{}", progress.epoch, progress.epoch_total)),
                ],
                vec![
                    title("Iteration :"),
                    value(format!("{}", progress.iteration)),
                ],
                vec![
                    title("Items     :"),
                    value(format!(
                        "{}/{}",
                        progress.progress.items_processed, progress.progress.items_total
                    )),
                ],
            ],
        }
    }

    pub(crate) fn render(self, frame: &mut TerminalFrame<'_>, size: Rect) {
        let paragraph = Paragraph::new(self.lines.into_iter().map(Line::from).collect::<Vec<_>>())
            .alignment(Alignment::Left)
            .block(Block::default().borders(Borders::ALL).title("Status"))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(Color::Gray));

        frame.render_widget(paragraph, size);
    }
}
