use std::rc::Rc;

use ratatui::{layout::{Layout, Direction, Constraint, Rect}, Frame};

pub(crate) struct Layouts {
    pub left: Rc<[Rect]>,
    pub right: Rc<[Rect]>,
}

impl Layouts {
    pub fn new(frame: &Frame) -> Self {
        let outer_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(vec![
                Constraint::Percentage(25),
                Constraint::Percentage(75),
            ])
            .split(frame.size());
        let left = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![
                Constraint::Percentage(40),
                Constraint::Percentage(40),
                Constraint::Percentage(20),
           ])
            .split(outer_layout[0]);
        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![
                Constraint::Percentage(80),
                Constraint::Percentage(20),
           ])
           .split(outer_layout[1]);
       Layouts {
           left,
           right,
       }
   }
}
