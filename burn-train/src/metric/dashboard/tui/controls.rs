use super::TFrame;
use ratatui::{
    prelude::{Alignment, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

pub(crate) struct ControlsView;

impl ControlsView {
    pub(crate) fn render<'b>(self, frame: &mut TFrame<'b>, size: Rect) {
        let lines = vec![
            vec![
                Span::from(" Quit  : ").yellow().bold(),
                Span::from("q  ").bold(),
                Span::from("  Stop the training gracefully").italic(),
            ],
            vec![
                Span::from(" Kill  : ").yellow().bold(),
                Span::from("k  ").bold(),
                Span::from("  Terminate the training").italic(),
            ],
            vec![
                Span::from(" Plots : ").yellow().bold(),
                Span::from("⬅ ➡").bold(),
                Span::from("  Switch between plots").italic(),
            ],
        ];
        let paragraph = Paragraph::new(
            lines
                .into_iter()
                .map(|span| Line::from(span))
                .collect::<Vec<_>>(),
        )
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(Color::Gray))
        .block(
            Block::default()
                .title_alignment(Alignment::Center)
                .borders(Borders::ALL)
                .style(Style::default().fg(Color::Gray))
                .title(Span::styled(
                    "Controls",
                    Style::default().add_modifier(Modifier::BOLD),
                )),
        );

        frame.render_widget(paragraph, size);
    }
}
