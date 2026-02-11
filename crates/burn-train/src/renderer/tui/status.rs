use crate::renderer::ProgressType;

use super::TerminalFrame;
use ratatui::{
    prelude::{Alignment, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

/// Show the training status with various information.
pub(crate) struct StatusState {
    progress_indicators: Vec<ProgressType>,
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
            progress_indicators: vec![],
            mode: Mode::Train,
        }
    }
}

impl StatusState {
    /// Update the training information.
    pub(crate) fn update_train(&mut self, progress_indicators: Vec<ProgressType>) {
        self.progress_indicators = progress_indicators;
        self.mode = Mode::Train;
    }
    /// Update the validation information.
    pub(crate) fn update_valid(&mut self, progress_indicators: Vec<ProgressType>) {
        self.progress_indicators = progress_indicators;
        self.mode = Mode::Valid;
    }
    /// Update the testing information.
    pub(crate) fn update_test(&mut self, progress_indicators: Vec<ProgressType>) {
        self.progress_indicators = progress_indicators;
        self.mode = Mode::Evaluation;
    }
    /// Create a view.
    pub(crate) fn view(&self) -> StatusView {
        StatusView::new(&self.progress_indicators, &self.mode)
    }
}

pub(crate) struct StatusView {
    lines: Vec<Vec<Span<'static>>>,
}

impl StatusView {
    fn new(progress_indicators: &[ProgressType], mode: &Mode) -> Self {
        let title = |title: &str| Span::from(format!(" {title} ")).bold().yellow();
        let value = |value: String| Span::from(value).italic();
        let mode = match mode {
            Mode::Valid => "Validating",
            Mode::Train => "Training",
            Mode::Evaluation => "Evaluation",
        };

        let width = progress_indicators
            .iter()
            .map(|p| match p {
                ProgressType::Detailed { tag, .. } => tag.len(),
                ProgressType::Value { tag, .. } => tag.len(),
            })
            .max()
            .unwrap_or(4);

        let mut lines = vec![vec![
            title(&format!("{: <width$} :", "Mode")),
            value(mode.to_string()),
        ]];

        progress_indicators.iter().for_each(|p| match p {
            ProgressType::Detailed { tag, progress } => lines.push(vec![
                title(&format!("{: <width$} :", tag)),
                value(format!(
                    "{}/{}",
                    progress.items_processed, progress.items_total
                )),
            ]),
            ProgressType::Value {
                tag,
                value: num_items,
            } => lines.push(vec![
                title(&format!("{: <width$} :", tag)),
                value(format!("{}", num_items)),
            ]),
        });

        Self { lines }
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
