no_analyze!{{
use centerLabels::*;let mut label = entry;
while let Some(next) = (|label| -> Option<centerLabels> { match label {
cl_tree_0 => {
if ({c+=1; c} >= w) { return None; }
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row11.add((c) as usize)).to_bool() {
						*img_labels_row00.add(c as usize) = *img_labels_row11.add((c) as usize);
						return Some(cl_tree_1);
					}
					else {
						*img_labels_row00.add(c as usize) = solver.new_label();
						return Some(cl_tree_1);
					}
				}
				else {
					*img_labels_row00.add(c as usize) = 0.elem();
					return Some(cl_tree_0);
				}
}
cl_tree_1 => {
if ({c+=1; c} >= w) { return None; }
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row11.add((c) as usize)).to_bool() {
						*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 1) as usize), *img_labels_row11.add((c) as usize), solver);
						return Some(cl_tree_1);
					}
					else {
						*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 1) as usize);
						return Some(cl_tree_1);
					}
				}
				else {
					*img_labels_row00.add(c as usize) = 0.elem();
					return Some(cl_tree_0);
				}
}
    }; None})(label)
{
label = next;
}
}}
