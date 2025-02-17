no_analyze!{{
use firstLabels::*;let mut label = entry;
while let Some(next) = (|label| -> Option<firstLabels> { match label {
		NODE_72=> {
		if (*img_row00.add((c + 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = solver.new_label();
			return Some(fl_tree_1);
		}
		else {
			*img_labels_row00.add(c as usize) = solver.new_label();
			return Some(fl_tree_2);
		}
				}
		NODE_73=> {
		if (*img_row00.add((c + 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
			return Some(fl_tree_1);
		}
		else {
			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
			return Some(fl_tree_2);
		}
				}
		NODE_74=> {
		if (*img_row00.add((c + 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = solver.new_label();
			return Some(fl_tree_1);
		}
		else {
			if (*img_row01.add((c + 1) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = solver.new_label();
				return Some(fl_tree_1);
			}
			else {
				*img_labels_row00.add(c as usize) = 0.elem();
				return Some(fl_tree_0);
			}
		}
				}
fl_tree_0 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(fl_break_0_0); } else { return Some(fl_break_1_0); } }
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_72);
				}
				else {
					if (*img_row01.add((c) as usize)).to_bool() {
						return Some(NODE_72);
					}
					else {
						return Some(NODE_74);
					}
				}
}
fl_tree_1 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(fl_break_0_1); } else { return Some(fl_break_1_1); } }
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_73);
				}
				else {
					if (*img_row01.add((c) as usize)).to_bool() {
						return Some(NODE_73);
					}
					else {
						return Some(NODE_74);
					}
				}
}
fl_tree_2 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(fl_break_0_2); } else { return Some(fl_break_1_2); } }
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row01.add((c - 1) as usize)).to_bool() {
						return Some(NODE_73);
					}
					else {
						return Some(NODE_72);
					}
				}
				else {
					if (*img_row01.add((c) as usize)).to_bool() {
						if (*img_row00.add((c + 1) as usize)).to_bool() {
							if (*img_row01.add((c - 1) as usize)).to_bool() {
								*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
								return Some(fl_tree_1);
							}
							else {
								*img_labels_row00.add(c as usize) = solver.new_label();
								return Some(fl_tree_1);
							}
						}
						else {
							if (*img_row01.add((c - 1) as usize)).to_bool() {
								*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
								return Some(fl_tree_2);
							}
							else {
								*img_labels_row00.add(c as usize) = solver.new_label();
								return Some(fl_tree_2);
							}
						}
					}
					else {
						return Some(NODE_74);
					}
				}
}
		NODE_75=> {
		if (*img_row01.add((c - 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
		}
		else {
			*img_labels_row00.add(c as usize) = solver.new_label();
		}
				}
fl_break_0_0 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					*img_labels_row00.add(c as usize) = solver.new_label();
				}
				else {
					if (*img_row01.add((c) as usize)).to_bool() {
						*img_labels_row00.add(c as usize) = solver.new_label();
					}
					else {
						*img_labels_row00.add(c as usize) = 0.elem();
					}
				}
		return None;}
fl_break_0_1 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
				}
				else {
					if (*img_row01.add((c) as usize)).to_bool() {
						*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
					}
					else {
						*img_labels_row00.add(c as usize) = 0.elem();
					}
				}
		return None;}
fl_break_0_2 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_75);
				}
				else {
					if (*img_row01.add((c) as usize)).to_bool() {
						return Some(NODE_75);
					}
					else {
						*img_labels_row00.add(c as usize) = 0.elem();
					}
				}
		return None;}
		NODE_76=> {
		if (*img_row00.add((c + 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = solver.new_label();
		}
		else {
			if (*img_row01.add((c + 1) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = solver.new_label();
			}
			else {
				*img_labels_row00.add(c as usize) = 0.elem();
			}
		}
				}
		NODE_77=> {
		if (*img_row01.add((c - 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
		}
		else {
			*img_labels_row00.add(c as usize) = solver.new_label();
		}
				}
fl_break_1_0 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					*img_labels_row00.add(c as usize) = solver.new_label();
				}
				else {
					if (*img_row01.add((c) as usize)).to_bool() {
						*img_labels_row00.add(c as usize) = solver.new_label();
					}
					else {
						return Some(NODE_76);
					}
				}
		return None;}
fl_break_1_1 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
				}
				else {
					if (*img_row01.add((c) as usize)).to_bool() {
						*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
					}
					else {
						return Some(NODE_76);
					}
				}
		return None;}
fl_break_1_2 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_77);
				}
				else {
					if (*img_row01.add((c) as usize)).to_bool() {
						if (*img_row00.add((c + 1) as usize)).to_bool() {
							return Some(NODE_77);
						}
						else {
							return Some(NODE_77);
						}
					}
					else {
						return Some(NODE_76);
					}
				}
		return None;}
fl_ => {},
    }; None})(label)
{
label = next;
}
}}
