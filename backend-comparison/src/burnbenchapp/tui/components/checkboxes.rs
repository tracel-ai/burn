use std::collections::BTreeSet;

use ratatui::{
    text::{Text, Line},
    widgets::{Paragraph, Scrollbar, StatefulWidget},
};

pub struct CheckboxesState {
    focused: usize,
    selected: BTreeSet<usize>,
}

impl Default for CheckboxesState {
    fn default() -> CheckboxesState {
        CheckboxesState {
            focused: 0,
            selected: BTreeSet::new(),
        }
    }
}

impl CheckboxesState {
    pub fn selected(&self) -> &BTreeSet<usize> {
        &self.selected
    }

    pub fn select(&mut self, index: usize) {
        self.selected.insert(index);
    }

    pub fn deselect(&mut self, index: usize) {
        self.selected.remove(&index);
    }

    pub fn deselect_all(&mut self) {
        self.selected.clear();
    }
}

pub struct Checkbox<'a> {
    content: &'a str,
    style: Style,
}

pub struct Checkboxes<'a> {
    items: Vec<Checkbox<'a>>,
    focused_style: Style,
    unfocused_style: Style,
    selected_symbol: &'a str,
    unselected_symbol: &'a str,
}

impl Checkboxes {
    pub fn new() -> Self {
        Checkboxes {
            items: [].to_vec(),
            focused_style: Style::default(),
            unfocused_style: Style::default(),
            selected_symbol: "☐ ",
            unselected_symbol: "☒ ",
        }
    }
}

impl StatefulWidget for Checkboxes {
    type State = CheckBoxesState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {

        let vertical_scroll = 0; // from app state
        let items = vec![
            Line::from("Item 1"),
            Line::from("Item 2"),
            Line::from("Item 3"),
        ];
        let pitems: Vec<Line> = self.items.iter().map(|i| Line::from(i)).collect();
        let paragraph = Paragraph::new(pitems.clone())
            .scroll((vertical_scroll as u16, 0))
            .block(Block::new().borders(Borders::RIGHT)); // to show a background for the scrollbar
        let scrollbar = Scrollbar::default()
            .orientation(ScrollbarOrientation::VerticalRight)
            .begin_symbol(Some("↑"))
            .end_symbol(Some("↓"));
        let mut scrollbar_state = ScrollbarState::new(items.iter().len()).position(vertical_scroll);
        let area = frame.size();
        frame.render_widget(paragraph, area);
        frame.render_stateful_widget(
            scrollbar,
            area.inner(&Margin {
                vertical: 1,
                horizontal: 0,
            }), // using a inner vertical margin of 1 unit makes the scrollbar inside the block
            &mut scrollbar_state,
        );
    }
}
