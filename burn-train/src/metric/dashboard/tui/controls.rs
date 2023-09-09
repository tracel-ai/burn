use super::TerminalFrame;
use ratatui::{
    prelude::{Alignment, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

/// Controls view.
pub(crate) struct ControlsView;

impl ControlsView {
    /// Render the view.
    pub(crate) fn render(self, frame: &mut TerminalFrame<'_>, size: Rect) {
        let lines = vec![
            vec![
                Span::from(" Quit          : ").yellow().bold(),
                Span::from("q  ").bold(),
                Span::from("  Stop the training.").italic(),
            ],
            vec![
                Span::from(" Plots Metrics : ").yellow().bold(),
                Span::from("⬅ ➡").bold(),
                Span::from("  Switch between metrics.").italic(),
            ],
            vec![
                Span::from(" Plots Type    : ").yellow().bold(),
                Span::from("⬆ ⬇").bold(),
                Span::from("  Switch between types.").italic(),
            ],
        ];
        let paragraph = Paragraph::new(lines.into_iter().map(Line::from).collect::<Vec<_>>())
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(Color::Gray))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .style(Style::default().fg(Color::Gray))
                    .title_alignment(Alignment::Left)
                    .title("Controls"),
            );

        frame.render_widget(paragraph, size);
    }
}
