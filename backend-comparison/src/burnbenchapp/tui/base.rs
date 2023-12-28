use std::{io, time::Duration};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    Terminal,
    prelude::Frame,
    buffer::Buffer,
    backend::CrosstermBackend,
    widgets::{StatefulWidget, Paragraph, Block, Borders, block::Position, Padding},
    style::Style,
    layout::{Alignment, Rect}
};

use crate::burnbenchapp::{
    tui::components::checkbox::*,
    Application,
};

type BenchTerminal = Terminal<CrosstermBackend<io::Stdout>>;

pub struct TuiApplication {
    terminal: BenchTerminal,
}

impl Application for TuiApplication {
    fn init(&mut self)  {
    }

    fn run(&mut self) {
        loop {
            self.terminal.draw(Self::render_app);
            if Self::should_quit() {
                break;
            }
        }
    }

    fn cleanup(&mut self) {
        disable_raw_mode();
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        self.terminal.show_cursor();
    }
}

impl TuiApplication {
    pub fn new() -> Self {
        TuiApplication {
            terminal: Self::setup_terminal(),
        }
    }


    fn setup_terminal() -> BenchTerminal {
        let mut stdout = io::stdout();
        enable_raw_mode().expect("enable terminal raw mode");
        execute!(stdout, EnterAlternateScreen);
        BenchTerminal::new(CrosstermBackend::new(stdout)).unwrap()
    }

    fn render_app(frame: &mut Frame) {
        let greeting = Paragraph::new("Hello World! (press 'q' to quit)");
        let block = Block::default()
            .title("Burn Bench")
            .title_position(Position::Top)
            .title_alignment(Alignment::Center)
            .borders(Borders::all())
            .border_style(Style::default().fg(ratatui::style::Color::DarkGray))
            .border_type(ratatui::widgets::BorderType::Double)
            .padding(Padding { left: 10, right: 10, top: 2, bottom: 2 })
            .style(Style::default().bg(ratatui::style::Color::Black));
        // frame.render_widget(greeting.block(block), frame.size());
        let checkbox = CustomCheckBox::new(String::from("My checkbox"));
        let mut checkbox_state = CustomCheckBoxState::default();
        frame.render_stateful_widget(checkbox, frame.size(), &mut checkbox_state);
    }

    fn should_quit() -> bool {
        if event::poll(Duration::from_millis(250)).unwrap() {
            if let Event::Key(key) = event::read().unwrap() {
                return KeyCode::Char('q')  == key.code
            }
        }
        false
    }

}
