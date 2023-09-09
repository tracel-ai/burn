use crossterm::event::{Event, KeyCode};
use ratatui::{
    prelude::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

use super::TerminalFrame;

/// Popup callback function.
pub(crate) trait CallbackFn: Send + Sync {
    /// Call the function and return if the popup state should be reset.
    fn call(&self) -> bool;
}

/// Popup callback.
pub(crate) struct Callback {
    title: String,
    description: String,
    trigger: char,
    callback: Box<dyn CallbackFn>,
}

impl Callback {
    /// Create a new popup.
    pub(crate) fn new<T, D, C>(title: T, description: D, trigger: char, callback: C) -> Self
    where
        T: Into<String>,
        D: Into<String>,
        C: CallbackFn + 'static,
    {
        Self {
            title: title.into(),
            description: description.into(),
            trigger,
            callback: Box::new(callback),
        }
    }
}

/// Popup state.
pub(crate) enum PopupState {
    Empty,
    Full(String, Vec<Callback>),
}

impl PopupState {
    /// If the popup is empty.
    pub(crate) fn is_empty(&self) -> bool {
        match &self {
            PopupState::Empty => true,
            _ => false,
        }
    }
    /// Handle popup events.
    pub(crate) fn on_event(&mut self, event: &Event) {
        let mut reset = false;

        match self {
            PopupState::Empty => {}
            PopupState::Full(_, callbacks) => {
                for callback in callbacks.iter() {
                    if let Event::Key(key) = event {
                        if let KeyCode::Char(key) = &key.code {
                            if &callback.trigger == key {
                                if callback.callback.call() {
                                    reset = true;
                                }
                            }
                        }
                    }
                }
            }
        };

        if reset {
            *self = Self::Empty;
        }
    }
    /// Create the popup view.
    pub(crate) fn view<'a>(&'a self) -> Option<PopupView<'a>> {
        match self {
            PopupState::Empty => None,
            PopupState::Full(title, callbacks) => Some(PopupView::new(&title, &callbacks)),
        }
    }
}

#[derive(new)]
pub(crate) struct PopupView<'a> {
    title: &'a String,
    callbacks: &'a [Callback],
}

impl<'a> PopupView<'a> {
    /// Render the view.
    pub(crate) fn render<'b>(&'a self, frame: &mut TerminalFrame<'b>, size: Rect) {
        let lines = self
            .callbacks
            .iter()
            .map(|callback| {
                vec![
                    Line::from(vec![
                        Span::from(format!("[{}] ", callback.trigger)).bold(),
                        Span::from(format!("{} ", callback.title)).yellow().bold(),
                    ]),
                    Line::from(Span::from("")),
                    Line::from(Span::from(format!("{}", callback.description)).italic()),
                    Line::from(Span::from("")),
                ]
            })
            .flatten()
            .collect::<Vec<_>>();

        let paragraph = Paragraph::new(lines)
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(Color::Gray))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title_alignment(Alignment::Center)
                    .style(Style::default().fg(Color::Gray))
                    .title(Span::styled(
                        self.title,
                        Style::default().add_modifier(Modifier::BOLD),
                    )),
            );

        let area = centered_rect(60, 40, size);
        frame.render_widget(paragraph, area);
    }
}

// From: https://github.com/ratatui-org/ratatui/blob/main/examples/popup.rs
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Percentage((100 - percent_y) / 2),
                Constraint::Percentage(percent_y),
                Constraint::Percentage((100 - percent_y) / 2),
            ]
            .as_ref(),
        )
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints(
            [
                Constraint::Percentage((100 - percent_x) / 2),
                Constraint::Percentage(percent_x),
                Constraint::Percentage((100 - percent_x) / 2),
            ]
            .as_ref(),
        )
        .split(popup_layout[1])[1]
}
