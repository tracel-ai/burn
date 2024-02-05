use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Margin},
    prelude::Frame,
    widgets::Paragraph,
    Terminal,
};
use std::{io, time::Duration};

use crate::burnbenchapp::{
    tui::components::regions::*, Application, BackendValues, BenchmarkValues,
};

type BenchTerminal = Terminal<CrosstermBackend<io::Stdout>>;

#[derive(PartialEq)]
enum Message {
    QuitApplication,
}

pub struct TuiApplication {
    terminal: BenchTerminal,
    regions: Regions<LeftRegion, RightRegion>,
}

impl Application for TuiApplication {
    fn init(&mut self) {}

    #[allow(unused)]
    fn run(&mut self, benches: &[BenchmarkValues], backends: &[BackendValues]) {
        // TODO initialize widgets given passed benches and backends on the command line
        loop {
            self.terminal
                .draw(|f| TuiApplication::render_app(&mut self.regions, f))
                .expect("frame should be drawn");
            let mut current_msg = self.handle_event();
            if let Some(Message::QuitApplication) = current_msg {
                break;
            } else {
                while current_msg.is_some() {
                    current_msg = self.update(current_msg.unwrap());
                }
            }
        }
    }

    fn cleanup(&mut self) {
        disable_raw_mode().expect("Terminal raw mode should be disabled");
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen)
            .expect("Alternate screen should be disabled");
        self.terminal
            .show_cursor()
            .expect("Terminal cursor should be made visible");
    }
}

impl TuiApplication {
    pub fn new() -> Self {
        TuiApplication {
            terminal: Self::setup_terminal(),
            regions: Regions::new(),
        }
    }

    fn setup_terminal() -> BenchTerminal {
        let mut stdout = io::stdout();
        enable_raw_mode().expect("Terminal raw mode should be enabled");
        execute!(stdout, EnterAlternateScreen).expect("Alternate screen should be enabled");
        BenchTerminal::new(CrosstermBackend::new(stdout)).unwrap()
    }

    fn handle_event(&mut self) -> Option<Message> {
        if event::poll(Duration::from_millis(250)).unwrap() {
            if let Event::Key(key) = event::read().unwrap() {
                match key.code {
                    KeyCode::Char('q') => return Some(Message::QuitApplication),
                    _ => {
                        self.regions.set_focus(key.code);
                    }
                }
            }
        }
        None
    }

    fn update(&mut self, _msg: Message) -> Option<Message> {
        None
    }

    fn render_app(regions: &mut Regions<LeftRegion, RightRegion>, frame: &mut Frame) {
        regions.draw(frame);
        let greeting =
            Paragraph::new("Work in Progress\n\n(press 'q' to quit)").alignment(Alignment::Center);
        frame.render_widget(
            greeting,
            regions.right.rect(&RightRegion::Top).inner(&Margin {
                horizontal: 1,
                vertical: 10,
            }),
        );
    }
}
