no_analyze!{{
use firstLabels::*;let mut label = entry;
while let Some(next) = (|label| -> Option<firstLabels> { match label {
fl_tree_0 => {
if ({c+=1; c} >= w) { return None; }
				if (*img_row00.add((c) as usize)).to_bool() {
					*img_labels_row00.add(c as usize) = solver.new_label();
					return Some(fl_tree_1);
				}
				else {
					*img_labels_row00.add(c as usize) = 0.elem();
					return Some(fl_tree_0);
				}
}
fl_tree_1 => {
if ({c+=1; c} >= w) { return None; }
				if (*img_row00.add((c) as usize)).to_bool() {
					*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 1) as usize);
					return Some(fl_tree_1);
				}
				else {
					*img_labels_row00.add(c as usize) = 0.elem();
					return Some(fl_tree_0);
				}
}
fl_ => {},
    }; None})(label)
{
label = next;
}
}}
