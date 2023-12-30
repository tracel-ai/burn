use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Margin},
    prelude::Frame,
    widgets::{Paragraph, ListState, List, Block, Borders},
    Terminal, style::{Style, Modifier},
};
use std::{io, time::Duration};

use crate::burnbenchapp::{tui::components::regions::*, Application};

type BenchTerminal = Terminal<CrosstermBackend<io::Stdout>>;

enum Message {
    QuitApplication,
}

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
            let mut current_msg = self.handle_event();
            while current_msg.is_some() {
                current_msg = self.update(current_msg.unwrap());
            }

            if Self::should_quit() {
                break;
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
        enable_raw_mode().expect("enable terminal raw mode");
        execute!(stdout, EnterAlternateScreen).expect("Alternate screen should be enabled");
        BenchTerminal::new(CrosstermBackend::new(stdout)).unwrap()
    }

    fn handle_event(&mut self) -> Option<Message> {
        if event::poll(Duration::from_millis(250)).unwrap() {
            if let Event::Key(key) = event::read().unwrap() {
                self.regions.set_focus(key.code);
            }
        }
        None
    }

    fn update(&mut self, msg: Message) -> Option<Message> {
        None
    }

    fn render_app(regions: &mut Regions<LeftRegion, RightRegion>, frame: &mut Frame) {
        regions.draw(frame);
        let greeting =
            Paragraph::new("Hello World! \n\n(press 'q' to quit)").alignment(Alignment::Center);
        frame.render_widget(
            greeting,
            regions.right.rect(&RightRegion::Top).inner(&Margin {
                horizontal: 1,
                vertical: 10,
            }),
        );
        let mut state = ListState::default();
        let items = ["Item 1", "Item 2", "Item 3"];
        let list = List::new(items)
            .block(Block::default())
            .highlight_style(Style::new().add_modifier(Modifier::REVERSED))
            .highlight_symbol(">>")
            .repeat_highlight_symbol(true);
        state.select(Some(0));
        frame.render_stateful_widget(
            list,
            regions.left.rect(&LeftRegion::Top).inner(&Margin { horizontal: 5, vertical: 1 }),
            &mut state);

        let mut state2 = ListState::default();
        let items2 = ["Item 1", "Item 2", "Item 3"];
        let list2 = List::new(items)
            .block(Block::default())
            .highlight_style(Style::new().add_modifier(Modifier::REVERSED))
            .highlight_symbol(">>")
            .repeat_highlight_symbol(true);
        state2.select(Some(1));
        frame.render_stateful_widget(
            list2,
            regions.left.rect(&LeftRegion::Middle).inner(&Margin { horizontal: 5, vertical: 1 }),
            &mut state2);


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
