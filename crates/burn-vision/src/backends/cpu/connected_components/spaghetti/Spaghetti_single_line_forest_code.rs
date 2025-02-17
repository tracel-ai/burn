no_analyze!{{
use singleLabels::*;let mut label = entry;
while let Some(next) = (|label| -> Option<singleLabels> { match label {
		NODE_93=> {
		if (*img_row00.add((c + 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = solver.new_label();
			return Some(sl_tree_1);
		}
		else {
			*img_labels_row00.add(c as usize) = 0.elem();
			return Some(sl_tree_0);
		}
				}
sl_tree_0 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(sl_break_0_0); } else { return Some(sl_break_1_0); } }
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row00.add((c + 1) as usize)).to_bool() {
						*img_labels_row00.add(c as usize) = solver.new_label();
						return Some(sl_tree_1);
					}
					else {
						*img_labels_row00.add(c as usize) = solver.new_label();
						return Some(sl_tree_0);
					}
				}
				else {
					return Some(NODE_93);
				}
}
sl_tree_1 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(sl_break_0_1); } else { return Some(sl_break_1_1); } }
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row00.add((c + 1) as usize)).to_bool() {
						*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
						return Some(sl_tree_1);
					}
					else {
						*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
						return Some(sl_tree_0);
					}
				}
				else {
					return Some(NODE_93);
				}
}
sl_break_0_0 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					*img_labels_row00.add(c as usize) = solver.new_label();
				}
				else {
					*img_labels_row00.add(c as usize) = 0.elem();
				}
		return None;}
sl_break_0_1 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
				}
				else {
					*img_labels_row00.add(c as usize) = 0.elem();
				}
		return None;}
		NODE_94=> {
		if (*img_row00.add((c + 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = solver.new_label();
		}
		else {
			*img_labels_row00.add(c as usize) = 0.elem();
		}
				}
sl_break_1_0 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					*img_labels_row00.add(c as usize) = solver.new_label();
				}
				else {
					return Some(NODE_94);
				}
		return None;}
sl_break_1_1 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
				}
				else {
					return Some(NODE_94);
				}
		return None;}
sl_ => {},
    }; None})(label)
{
label = next;
}
}}
