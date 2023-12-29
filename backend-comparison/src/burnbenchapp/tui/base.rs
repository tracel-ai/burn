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

use crate::burnbenchapp::{tui::components::regions::*, Application};

type BenchTerminal = Terminal<CrosstermBackend<io::Stdout>>;

pub struct TuiApplication {
    terminal: BenchTerminal,
    regions: Regions<LeftRegion, RightRegion>,
}

impl Application for TuiApplication {
    fn init(&mut self) {}

    fn run(&mut self) {
        loop {
            self.terminal
                .draw(|f| TuiApplication::render_app(&mut self.regions, f))
                .expect("frame should be drawn");
            if Self::should_quit() {
                break;
            }
        }
    }

    fn cleanup(&mut self) {
        disable_raw_mode().expect("Terminal raw mode should be disabled");
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen)
            .expect("Alternate screen should be disabled");
        self.terminal.show_cursor().expect("Terminal cursor should be made visible");
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
        enable_raw_mode().expect("enable terminal raw mode");
        execute!(stdout, EnterAlternateScreen).expect("Alternate screen should be enabled");
        BenchTerminal::new(CrosstermBackend::new(stdout)).unwrap()
    }

    // fn handle_event() -> Result<Option<Message>, String> {
    //     if event::poll(Duration::from_millis(250)).unwrap() {
    //         if let Event::Key(key) = event::read()? {
    //             if key.kind == event::KeyEventKind::Press {
    //                 return Ok(handle_key(key));
    //             }
    //         }
    //     }
    // }

    // fn handle_focus_key(key: event::KeyEvent) -> Option<FocusMessage> {
    //     match key.code {
    //         KeyCode::Char(c) => {
    //             if c == LeftRegion::Top.get_rect_info().hotkey {
    //                 Some()
    //             } else if c == LeftRegion::Middle.get_rect_info().hotkey {
    //                 Some()
    //             } else if c == LeftRegion::Bottom.get_rect_info().hotkey {

    //                 Some()
    //             } else if c == RightRegion::Top.get_rect_info().hotkey {
    //                 Some()
    //             } else if c == RightRegion::Bottom.get_rect_info().hotkey {
    //                 Some()
    //             } else {
    //                 None
    //             }
    //         },
    //         _ => None,
    //     }
    // }

    fn render_app(regions: &mut Regions<LeftRegion, RightRegion>, frame: &mut Frame) {
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
