use ratatui::{
    crossterm::event::{Event, KeyCode},
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
        matches!(&self, PopupState::Empty)
    }
    /// Handle popup events.
    pub(crate) fn on_event(&mut self, event: &Event) {
        let mut reset = false;

        match self {
            PopupState::Empty => {}
            PopupState::Full(_, callbacks) => {
                for callback in callbacks.iter() {
                    if let Event::Key(key) = event
                        && let KeyCode::Char(key) = &key.code
                        && &callback.trigger == key
                        && callback.callback.call()
                    {
                        reset = true;
                    }
                }
            }
        };

        if reset {
            *self = Self::Empty;
        }
    }
    /// Create the popup view.
    pub(crate) fn view(&self) -> Option<PopupView<'_>> {
        match self {
            PopupState::Empty => None,
            PopupState::Full(title, callbacks) => Some(PopupView::new(title, callbacks)),
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
            .flat_map(|callback| {
                vec![
                    Line::from(vec![
                        Span::from(format!("[{}] ", callback.trigger)).bold(),
                        Span::from(format!("{} ", callback.title)).yellow().bold(),
                    ]),
                    Line::from(Span::from("")),
                    Line::from(Span::from(callback.description.to_string()).italic()),
                    Line::from(Span::from("")),
                ]
            })
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

        let area = centered_percent(20, size, Direction::Horizontal);
        let area = centered_percent(20, area, Direction::Vertical);

        frame.render_widget(paragraph, area);
    }
}

/// The percent represents the amount of space that will be taken by each side.
fn centered_percent(percent: u16, size: Rect, direction: Direction) -> Rect {
    let center = 100 - (percent * 2);

    Layout::default()
        .direction(direction)
        .constraints([
            Constraint::Percentage(percent),
            Constraint::Percentage(center),
            Constraint::Percentage(percent),
        ])
        .split(size)[1]
}
