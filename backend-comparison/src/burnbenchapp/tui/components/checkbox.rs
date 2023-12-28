use ratatui::{widgets::StatefulWidget, layout::Rect, buffer::Buffer, style::Style};

#[derive(Default)]
pub(crate) struct CustomCheckBoxState {
    checked: bool,
}

#[derive(new)]
pub(crate) struct CustomCheckBox {
    label: String,
}

impl StatefulWidget for CustomCheckBox {
    // Define the state used by the custom checkbox
    type State = CustomCheckBoxState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        // Define the look of the checkbox based on the state
        let checkbox_char = if state.checked { "☒" } else { "☐" };

        // Write the checkbox and label to the buffer
        buf.set_string(
            area.x,
            area.y,
            format!("{} {}", checkbox_char, self.label),
            Style::default()
        );
    }
}
