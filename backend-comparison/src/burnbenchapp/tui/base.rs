use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    buffer::Buffer,
    layout::{Alignment, Direction, Layout, Margin, Rect},
    prelude::Frame,
    style::Style,
    widgets::{block::Position, Block, Borders, Padding, Paragraph, StatefulWidget},
    Terminal,
};
use std::{io, time::Duration};

use crate::burnbenchapp::{
    tui::components::{checkbox::*, regions::*},
    Application,
};

type BenchTerminal = Terminal<CrosstermBackend<io::Stdout>>;

pub struct TuiApplication {
    terminal: BenchTerminal,
}

impl Application for TuiApplication {
    fn init(&mut self) {}

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
        let regions = Regions::new(frame);
        regions.draw(frame);
        let greeting =
            Paragraph::new("Hello World! (press 'q' to quit)").alignment(Alignment::Center);
        frame.render_widget(
            greeting,
            regions.right.rect(RightRegion::Top).inner(&Margin {
                horizontal: 1,
                vertical: 10,
            }),
        );
        // let checkbox = CustomCheckBox::new(String::from("My checkbox"));
        // let mut checkbox_state = CustomCheckBoxState::default();
        // frame.render_stateful_widget(
        //     checkbox,
        //     regions.right.get_rect(RightRegionPosition::Bottom),
        //     &mut checkbox_state,
        // );
    }

    fn should_quit() -> bool {
        if event::poll(Duration::from_millis(250)).unwrap() {
            if let Event::Key(key) = event::read().unwrap() {
                return KeyCode::Char('q') == key.code;
            }
        }
        false
    }
}
