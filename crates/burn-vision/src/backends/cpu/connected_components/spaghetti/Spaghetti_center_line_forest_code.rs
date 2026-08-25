no_analyze!{{
use centerLabels::{NODE_1, NODE_2, NODE_3, cl_tree_2, cl_tree_1, NODE_4, NODE_5, cl_tree_5, cl_tree_4, cl_tree_3, NODE_6, cl_tree_7, cl_tree_6, NODE_7, NODE_8, NODE_9, NODE_10, cl_tree_11, cl_tree_8, NODE_11, cl_tree_12, NODE_12, NODE_13, cl_tree_10, cl_tree_9, NODE_14, NODE_15, NODE_16, NODE_17, NODE_18, NODE_19, NODE_20, NODE_21, NODE_22, NODE_23, NODE_24, NODE_25, NODE_26, NODE_27, NODE_28, NODE_29, NODE_30, NODE_31, NODE_32, NODE_33, NODE_34, NODE_35, NODE_36, NODE_37, NODE_38, NODE_39, NODE_40, cl_tree_0, cl_break_0_0, cl_break_1_0, cl_break_0_1, cl_break_1_1, cl_break_0_2, cl_break_1_2, cl_break_0_3, cl_break_1_3, cl_break_1_4, cl_break_1_5, cl_break_1_6, cl_break_0_4, cl_break_1_7, cl_break_1_8, cl_break_0_5, cl_break_1_9, cl_break_0_6, cl_break_1_10, cl_break_0_7, cl_break_1_11, cl_break_0_8, cl_break_1_12, NODE_41, NODE_42, NODE_43, NODE_44, NODE_45, NODE_46, NODE_47, NODE_48, NODE_49, NODE_50, NODE_51, NODE_52, NODE_53, NODE_54, NODE_55, NODE_56, NODE_57, NODE_58, NODE_59, NODE_60, NODE_61, NODE_62, NODE_63, NODE_64, NODE_65, NODE_66, NODE_67, NODE_68, NODE_69, NODE_70, NODE_71};let mut label = entry;
while let Some(next) = (|label| -> Option<centerLabels> { match label {
		NODE_1=> {
		if (*img_row00.add((c + 1) as usize)).to_bool() {
		return Some(NODE_2);
		}
  		return Some(NODE_3);
				}
		NODE_3=> {
		if (*img_row01.add((c + 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = solver.new_label();
			return Some(cl_tree_2);
		}
  			*img_labels_row00.add(c as usize) = 0.elem();
  			return Some(cl_tree_1);
				}
		NODE_4=> {
		if (*img_row11.add((c + 2) as usize)).to_bool() {
			if (*img_row11.add((c) as usize)).to_bool() {
			return Some(NODE_5);
			}
   				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c + 2) as usize);
   				return Some(cl_tree_5);
		}
  			if (*img_row11.add((c) as usize)).to_bool() {
  				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
  				return Some(cl_tree_4);
  			}
     				*img_labels_row00.add(c as usize) = solver.new_label();
     				return Some(cl_tree_3);
				}
		NODE_6=> {
		if (*img_row01.add((c) as usize)).to_bool() {
			if (*img_row00.add((c + 1) as usize)).to_bool() {
			return Some(NODE_2);
			}
   				*img_labels_row00.add(c as usize) = solver.new_label();
   				return Some(cl_tree_7);
		}
  		return Some(NODE_1);
				}
		NODE_2=> {
		if (*img_row11.add((c + 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
			return Some(cl_tree_6);
		}
  		return Some(NODE_4);
				}
		NODE_7=> {
		if (*img_row12.add((c + 1) as usize)).to_bool() {
			if (*img_row12.add((c) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c + 2) as usize);
				return Some(cl_tree_5);
			}
   				*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row12.add((c + 2) as usize), *img_labels_row12.add((c - 2) as usize), solver);
   				return Some(cl_tree_5);
		}
  			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row12.add((c + 2) as usize), *img_labels_row12.add((c - 2) as usize), solver);
  			return Some(cl_tree_5);
				}
		NODE_5=> {
		if (*img_row12.add((c + 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c + 2) as usize);
			return Some(cl_tree_5);
		}
  			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row12.add((c + 2) as usize), *img_labels_row12.add((c) as usize), solver);
  			return Some(cl_tree_5);
				}
		NODE_8=> {
		if (*img_row11.add((c + 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
			return Some(cl_tree_6);
		}
  		return Some(NODE_9);
				}
		NODE_10=> {
		if (*img_row11.add((c + 1) as usize)).to_bool() {
			if (*img_row11.add((c) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
				return Some(cl_tree_11);
			}
   				if (*img_row11.add((c - 1) as usize)).to_bool() {
   					if (*img_row12.add((c) as usize)).to_bool() {
   						*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
   						return Some(cl_tree_11);
   					}
        						*img_labels_row00.add(c as usize) = LabelsSolver::merge(LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver), *img_labels_row12.add((c - 2) as usize), solver);
        						return Some(cl_tree_11);
   				}
       					*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
       					return Some(cl_tree_11);
		}
  			if (*img_row00.add((c + 1) as usize)).to_bool() {
  				if (*img_row11.add((c + 2) as usize)).to_bool() {
  					if (*img_row11.add((c) as usize)).to_bool() {
  						if (*img_row12.add((c + 1) as usize)).to_bool() {
  							*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
  							return Some(cl_tree_5);
  						}
        							*img_labels_row00.add(c as usize) = LabelsSolver::merge(LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver), *img_labels_row12.add((c) as usize), solver);
        							return Some(cl_tree_5);
  					}
       						if (*img_row11.add((c - 1) as usize)).to_bool() {
       							if (*img_row12.add((c + 1) as usize)).to_bool() {
       								if (*img_row12.add((c) as usize)).to_bool() {
       									*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
       									return Some(cl_tree_5);
       								}
               									*img_labels_row00.add(c as usize) = LabelsSolver::merge(LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver), *img_labels_row12.add((c - 2) as usize), solver);
               									return Some(cl_tree_5);
       							}
              								*img_labels_row00.add(c as usize) = LabelsSolver::merge(LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver), *img_labels_row12.add((c - 2) as usize), solver);
              								return Some(cl_tree_5);
       						}
             							*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
             							return Some(cl_tree_5);
  				}
      					if (*img_row11.add((c - 1) as usize)).to_bool() {
      						*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c - 2) as usize), solver);
      						return Some(cl_tree_8);
      					}
           					return Some(NODE_11);
  			}
     				if (*img_row11.add((c - 1) as usize)).to_bool() {
     					*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c - 2) as usize), solver);
     					return Some(cl_tree_12);
     				}
         				return Some(NODE_12);
				}
		NODE_11=> {
		if (*img_row11.add((c) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
			return Some(cl_tree_4);
		}
  			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
  			return Some(cl_tree_3);
				}
		NODE_13=> {
		if (*img_row12.add((c) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
			return Some(cl_tree_11);
		}
  			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row12.add((c) as usize), *img_labels_row12.add((c - 2) as usize), solver);
  			return Some(cl_tree_11);
				}
		NODE_9=> {
		if (*img_row11.add((c + 2) as usize)).to_bool() {
			if (*img_row12.add((c + 1) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
				return Some(cl_tree_5);
			}
   				if (*img_row11.add((c) as usize)).to_bool() {
   					*img_labels_row00.add(c as usize) = LabelsSolver::merge(LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver), *img_labels_row12.add((c) as usize), solver);
   					return Some(cl_tree_5);
   				}
       					*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
       					return Some(cl_tree_5);
		}
  		return Some(NODE_11);
				}
		NODE_12=> {
		if (*img_row11.add((c) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
			return Some(cl_tree_10);
		}
  			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
  			return Some(cl_tree_9);
				}
		NODE_14=> {
		if (*img_row11.add((c + 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
			return Some(cl_tree_11);
		}
  			if (*img_row00.add((c + 1) as usize)).to_bool() {
  			return Some(NODE_4);
  			}
     				if (*img_row11.add((c) as usize)).to_bool() {
     					*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
     					return Some(cl_tree_10);
     				}
         					*img_labels_row00.add(c as usize) = solver.new_label();
         					return Some(cl_tree_9);
				}
		NODE_15=> {
		if (*img_row11.add((c + 1) as usize)).to_bool() {
			if (*img_row11.add((c) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
				return Some(cl_tree_11);
			}
   				if (*img_row11.add((c - 1) as usize)).to_bool() {
   				return Some(NODE_13);
   				}
       					*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
       					return Some(cl_tree_11);
		}
  			if (*img_row00.add((c + 1) as usize)).to_bool() {
  				if (*img_row11.add((c + 2) as usize)).to_bool() {
  					if (*img_row11.add((c) as usize)).to_bool() {
  					return Some(NODE_5);
  					}
       						if (*img_row11.add((c - 1) as usize)).to_bool() {
       						return Some(NODE_7);
       						}
             							*img_labels_row00.add(c as usize) = *img_labels_row12.add((c + 2) as usize);
             							return Some(cl_tree_5);
  				}
      					if (*img_row11.add((c) as usize)).to_bool() {
      						*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
      						return Some(cl_tree_4);
      					}
           						if (*img_row11.add((c - 1) as usize)).to_bool() {
           							*img_labels_row00.add(c as usize) = *img_labels_row12.add((c - 2) as usize);
           							return Some(cl_tree_3);
           						}
                 							*img_labels_row00.add(c as usize) = solver.new_label();
                 							return Some(cl_tree_3);
  			}
     				if (*img_row11.add((c) as usize)).to_bool() {
     					*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
     					return Some(cl_tree_10);
     				}
         					if (*img_row11.add((c - 1) as usize)).to_bool() {
         						*img_labels_row00.add(c as usize) = *img_labels_row12.add((c - 2) as usize);
         						return Some(cl_tree_9);
         					}
              						*img_labels_row00.add(c as usize) = solver.new_label();
              						return Some(cl_tree_9);
				}
		NODE_16=> {
		if (*img_row01.add((c) as usize)).to_bool() {
			if (*img_row00.add((c + 1) as usize)).to_bool() {
				if (*img_row01.add((c - 1) as usize)).to_bool() {
				return Some(NODE_8);
				}
    				return Some(NODE_2);
			}
   			return Some(NODE_17);
		}
  		return Some(NODE_1);
				}
		NODE_18=> {
		if (*img_row12.add((c - 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c + 2) as usize);
			return Some(cl_tree_5);
		}
  			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
  			return Some(cl_tree_5);
				}
		NODE_19=> {
		if (*img_row11.add((c + 2) as usize)).to_bool() {
		return Some(NODE_20);
		}
  			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
  			return Some(cl_tree_8);
				}
		NODE_21=> {
		if (*img_row00.add((c + 1) as usize)).to_bool() {
			if (*img_row11.add((c + 1) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
				return Some(cl_tree_6);
			}
   				if (*img_row11.add((c + 2) as usize)).to_bool() {
   					*img_labels_row00.add(c as usize) = *img_labels_row12.add((c + 2) as usize);
   					return Some(cl_tree_5);
   				}
       					*img_labels_row00.add(c as usize) = solver.new_label();
       					return Some(cl_tree_3);
		}
  		return Some(NODE_3);
				}
		NODE_22=> {
		if (*img_row11.add((c) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
			return Some(cl_tree_6);
		}
  			if (*img_row12.add((c) as usize)).to_bool() {
  				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
  				return Some(cl_tree_6);
  			}
     				*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
     				return Some(cl_tree_6);
				}
		NODE_23=> {
		if (*img_row11.add((c) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
			return Some(cl_tree_11);
		}
  			if (*img_row12.add((c) as usize)).to_bool() {
  				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
  				return Some(cl_tree_11);
  			}
     				*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
     				return Some(cl_tree_11);
				}
		NODE_24=> {
		if (*img_row12.add((c - 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
			return Some(cl_tree_6);
		}
  			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
  			return Some(cl_tree_6);
				}
		NODE_17=> {
		if (*img_row01.add((c - 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
			return Some(cl_tree_7);
		}
  			*img_labels_row00.add(c as usize) = solver.new_label();
  			return Some(cl_tree_7);
				}
		NODE_25=> {
		if (*img_row11.add((c + 2) as usize)).to_bool() {
			if (*img_row12.add((c + 1) as usize)).to_bool() {
				if (*img_row12.add((c) as usize)).to_bool() {
				return Some(NODE_18);
				}
    					*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
    					return Some(cl_tree_5);
			}
   				*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
   				return Some(cl_tree_5);
		}
  			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
  			return Some(cl_tree_8);
				}
		NODE_20=> {
		if (*img_row12.add((c + 1) as usize)).to_bool() {
		return Some(NODE_26);
		}
  			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
  			return Some(cl_tree_5);
				}
		NODE_27=> {
		if (*img_row12.add((c - 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
			return Some(cl_tree_11);
		}
  			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
  			return Some(cl_tree_11);
				}
		NODE_28=> {
		if (*img_row11.add((c + 1) as usize)).to_bool() {
		return Some(NODE_22);
		}
  		return Some(NODE_19);
				}
		NODE_26=> {
		if (*img_row11.add((c) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c + 2) as usize);
			return Some(cl_tree_5);
		}
  			if (*img_row12.add((c) as usize)).to_bool() {
  				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c + 2) as usize);
  				return Some(cl_tree_5);
  			}
     				*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
     				return Some(cl_tree_5);
				}
		NODE_29=> {
		if (*img_row11.add((c + 2) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
			return Some(cl_tree_5);
		}
  			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
  			return Some(cl_tree_8);
				}
		NODE_30=> {
		if (*img_row11.add((c + 2) as usize)).to_bool() {
			if (*img_row12.add((c + 1) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c + 2) as usize);
				return Some(cl_tree_5);
			}
   				*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
   				return Some(cl_tree_5);
		}
  			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
  			return Some(cl_tree_8);
				}
		NODE_31=> {
		if (*img_row11.add((c + 1) as usize)).to_bool() {
		return Some(NODE_23);
		}
  			if (*img_row00.add((c + 1) as usize)).to_bool() {
  			return Some(NODE_19);
  			}
     				*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
     				return Some(cl_tree_12);
				}
		NODE_32=> {
		if (*img_row11.add((c + 2) as usize)).to_bool() {
			if (*img_row12.add((c + 1) as usize)).to_bool() {
				if (*img_row11.add((c - 2) as usize)).to_bool() {
				return Some(NODE_33);
				}
    					*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
    					return Some(cl_tree_5);
			}
   				if (*img_row11.add((c) as usize)).to_bool() {
   					if (*img_row11.add((c - 2) as usize)).to_bool() {
   					return Some(NODE_34);
   					}
        						*img_labels_row00.add(c as usize) = LabelsSolver::merge(LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver), *img_labels_row12.add((c) as usize), solver);
        						return Some(cl_tree_5);
   				}
       					*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
       					return Some(cl_tree_5);
		}
  			if (*img_row11.add((c) as usize)).to_bool() {
  				if (*img_row11.add((c - 2) as usize)).to_bool() {
  				return Some(NODE_35);
  				}
      					*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
      					return Some(cl_tree_4);
  			}
     				*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
     				return Some(cl_tree_3);
				}
		NODE_36=> {
		if (*img_row11.add((c + 2) as usize)).to_bool() {
			if (*img_row12.add((c + 1) as usize)).to_bool() {
			return Some(NODE_33);
			}
   				if (*img_row11.add((c) as usize)).to_bool() {
   				return Some(NODE_34);
   				}
       					*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
       					return Some(cl_tree_5);
		}
  			if (*img_row11.add((c) as usize)).to_bool() {
  			return Some(NODE_35);
  			}
     				*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
     				return Some(cl_tree_3);
				}
		NODE_37=> {
		if (*img_row11.add((c + 2) as usize)).to_bool() {
			if (*img_row12.add((c + 1) as usize)).to_bool() {
				if (*img_row12.add((c) as usize)).to_bool() {
					if (*img_row11.add((c - 2) as usize)).to_bool() {
					return Some(NODE_18);
					}
     						*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
     						return Some(cl_tree_5);
				}
    					*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
    					return Some(cl_tree_5);
			}
   				*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
   				return Some(cl_tree_5);
		}
  			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
  			return Some(cl_tree_8);
				}
		NODE_33=> {
		if (*img_row12.add((c - 1) as usize)).to_bool() {
		return Some(NODE_26);
		}
  			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
  			return Some(cl_tree_5);
				}
		NODE_38=> {
		if (*img_row12.add((c - 1) as usize)).to_bool() {
		return Some(NODE_22);
		}
  			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
  			return Some(cl_tree_6);
				}
		NODE_39=> {
		if (*img_row12.add((c - 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
			return Some(cl_tree_10);
		}
  			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
  			return Some(cl_tree_10);
				}
		NODE_35=> {
		if (*img_row12.add((c - 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
			return Some(cl_tree_4);
		}
  			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
  			return Some(cl_tree_4);
				}
		NODE_40=> {
		if (*img_row12.add((c - 1) as usize)).to_bool() {
		return Some(NODE_23);
		}
  			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
  			return Some(cl_tree_11);
				}
		NODE_34=> {
		if (*img_row12.add((c - 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row12.add((c + 2) as usize), *img_labels_row12.add((c) as usize), solver);
			return Some(cl_tree_5);
		}
  			*img_labels_row00.add(c as usize) = LabelsSolver::merge(LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver), *img_labels_row12.add((c) as usize), solver);
  			return Some(cl_tree_5);
				}
cl_tree_0 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(cl_break_0_0); }return Some(cl_break_1_0); }
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_14);
				}
    					return Some(NODE_6);
}
cl_tree_1 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(cl_break_0_1); }return Some(cl_break_1_1); }
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_15);
				}
    					return Some(NODE_6);
}
cl_tree_2 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(cl_break_0_2); }return Some(cl_break_1_2); }
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_10);
				}
    					if (*img_row01.add((c) as usize)).to_bool() {
    						if (*img_row00.add((c + 1) as usize)).to_bool() {
    							return Some(NODE_8);
    						}
          							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
          							return Some(cl_tree_7);
    					}
         						return Some(NODE_1);
}
cl_tree_3 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(cl_break_0_3); }return Some(cl_break_1_3); }
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row11.add((c + 1) as usize)).to_bool() {
						*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
						return Some(cl_tree_11);
					}
     						if (*img_row00.add((c + 1) as usize)).to_bool() {
     							return Some(NODE_29);
     						}
           							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
           							return Some(cl_tree_12);
				}
    					if (*img_row01.add((c) as usize)).to_bool() {
    						if (*img_row00.add((c + 1) as usize)).to_bool() {
    							if (*img_row11.add((c + 1) as usize)).to_bool() {
    								*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
    								return Some(cl_tree_6);
    							}
           								return Some(NODE_29);
    						}
          							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
          							return Some(cl_tree_7);
    					}
         						return Some(NODE_21);
}
cl_tree_4 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(cl_break_0_3); }return Some(cl_break_1_4); }
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row11.add((c + 1) as usize)).to_bool() {
						if (*img_row12.add((c) as usize)).to_bool() {
							return Some(NODE_27);
						}
      							*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
      							return Some(cl_tree_11);
					}
     						if (*img_row00.add((c + 1) as usize)).to_bool() {
     							return Some(NODE_25);
     						}
           							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
           							return Some(cl_tree_12);
				}
    					if (*img_row01.add((c) as usize)).to_bool() {
    						if (*img_row00.add((c + 1) as usize)).to_bool() {
    							if (*img_row11.add((c + 1) as usize)).to_bool() {
    								if (*img_row12.add((c) as usize)).to_bool() {
    									return Some(NODE_24);
    								}
            									*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
            									return Some(cl_tree_6);
    							}
           								return Some(NODE_25);
    						}
          							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
          							return Some(cl_tree_7);
    					}
         						return Some(NODE_21);
}
cl_tree_5 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(cl_break_0_3); }return Some(cl_break_1_5); }
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row11.add((c + 1) as usize)).to_bool() {
						*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
						return Some(cl_tree_11);
					}
     						if (*img_row00.add((c + 1) as usize)).to_bool() {
     							return Some(NODE_30);
     						}
           							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
           							return Some(cl_tree_12);
				}
    					if (*img_row01.add((c) as usize)).to_bool() {
    						if (*img_row00.add((c + 1) as usize)).to_bool() {
    							if (*img_row11.add((c + 1) as usize)).to_bool() {
    								*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
    								return Some(cl_tree_6);
    							}
           								return Some(NODE_30);
    						}
          							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
          							return Some(cl_tree_7);
    					}
         						if (*img_row00.add((c + 1) as usize)).to_bool() {
         							if (*img_row11.add((c + 1) as usize)).to_bool() {
         								*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
         								return Some(cl_tree_6);
         							}
                								if (*img_row11.add((c + 2) as usize)).to_bool() {
                									return Some(NODE_5);
                								}
                        									*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
                        									return Some(cl_tree_4);
         						}
               							return Some(NODE_3);
}
cl_tree_6 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(cl_break_0_3); }return Some(cl_break_1_6); }
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_31);
				}
    					if (*img_row01.add((c) as usize)).to_bool() {
    						if (*img_row00.add((c + 1) as usize)).to_bool() {
    							return Some(NODE_28);
    						}
          							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
          							return Some(cl_tree_7);
    					}
         						return Some(NODE_1);
}
cl_tree_7 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(cl_break_0_4); }return Some(cl_break_1_7); }
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row01.add((c - 1) as usize)).to_bool() {
						return Some(NODE_10);
					}
     						return Some(NODE_15);
				}
    					return Some(NODE_16);
}
cl_tree_8 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(cl_break_0_3); }return Some(cl_break_1_8); }
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row11.add((c + 1) as usize)).to_bool() {
						if (*img_row12.add((c) as usize)).to_bool() {
							if (*img_row11.add((c - 2) as usize)).to_bool() {
								return Some(NODE_27);
							}
       								*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
       								return Some(cl_tree_11);
						}
      							*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
      							return Some(cl_tree_11);
					}
     						if (*img_row00.add((c + 1) as usize)).to_bool() {
     							return Some(NODE_37);
     						}
           							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
           							return Some(cl_tree_12);
				}
    					if (*img_row01.add((c) as usize)).to_bool() {
    						if (*img_row00.add((c + 1) as usize)).to_bool() {
    							if (*img_row11.add((c + 1) as usize)).to_bool() {
    								if (*img_row12.add((c) as usize)).to_bool() {
    									if (*img_row11.add((c - 2) as usize)).to_bool() {
    										return Some(NODE_24);
    									}
             										*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
             										return Some(cl_tree_6);
    								}
            									*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
            									return Some(cl_tree_6);
    							}
           								return Some(NODE_37);
    						}
          							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
          							return Some(cl_tree_7);
    					}
         						return Some(NODE_21);
}
cl_tree_9 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(cl_break_0_5); }return Some(cl_break_1_9); }
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row01.add((c - 1) as usize)).to_bool() {
						if (*img_row11.add((c + 1) as usize)).to_bool() {
							*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
							return Some(cl_tree_11);
						}
      							if (*img_row00.add((c + 1) as usize)).to_bool() {
      								return Some(NODE_9);
      							}
             								return Some(NODE_12);
					}
     						return Some(NODE_14);
				}
    					return Some(NODE_16);
}
cl_tree_10 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(cl_break_0_6); }return Some(cl_break_1_10); }
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row01.add((c - 1) as usize)).to_bool() {
						if (*img_row11.add((c + 1) as usize)).to_bool() {
							return Some(NODE_40);
						}
      							if (*img_row00.add((c + 1) as usize)).to_bool() {
      								return Some(NODE_36);
      							}
             								if (*img_row11.add((c) as usize)).to_bool() {
             									return Some(NODE_39);
             								}
                     									*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
                     									return Some(cl_tree_9);
					}
     						return Some(NODE_14);
				}
    					if (*img_row01.add((c) as usize)).to_bool() {
    						if (*img_row00.add((c + 1) as usize)).to_bool() {
    							if (*img_row01.add((c - 1) as usize)).to_bool() {
    								if (*img_row11.add((c + 1) as usize)).to_bool() {
    									return Some(NODE_38);
    								}
            									return Some(NODE_36);
    							}
           								return Some(NODE_2);
    						}
          							return Some(NODE_17);
    					}
         						return Some(NODE_1);
}
cl_tree_11 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(cl_break_0_7); }return Some(cl_break_1_11); }
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row00.add((c - 1) as usize)).to_bool() {
						return Some(NODE_31);
					}
     						if (*img_row01.add((c - 1) as usize)).to_bool() {
     							return Some(NODE_31);
     						}
           							if (*img_row11.add((c + 1) as usize)).to_bool() {
           								if (*img_row11.add((c) as usize)).to_bool() {
           									*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
           									return Some(cl_tree_11);
           								}
                   									return Some(NODE_13);
           							}
                  								if (*img_row00.add((c + 1) as usize)).to_bool() {
                  									if (*img_row11.add((c + 2) as usize)).to_bool() {
                  										if (*img_row11.add((c) as usize)).to_bool() {
                  											return Some(NODE_5);
                  										}
                            											return Some(NODE_7);
                  									}
                           										if (*img_row11.add((c) as usize)).to_bool() {
                           											*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
                           											return Some(cl_tree_4);
                           										}
                                     											*img_labels_row00.add(c as usize) = *img_labels_row12.add((c - 2) as usize);
                                     											return Some(cl_tree_3);
                  								}
                          									if (*img_row11.add((c) as usize)).to_bool() {
                          										*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
                          										return Some(cl_tree_10);
                          									}
                                   										*img_labels_row00.add(c as usize) = *img_labels_row12.add((c - 2) as usize);
                                   										return Some(cl_tree_9);
				}
    					if (*img_row01.add((c) as usize)).to_bool() {
    						if (*img_row00.add((c + 1) as usize)).to_bool() {
    							if (*img_row00.add((c - 1) as usize)).to_bool() {
    								return Some(NODE_28);
    							}
           								if (*img_row01.add((c - 1) as usize)).to_bool() {
           									if (*img_row11.add((c + 1) as usize)).to_bool() {
           										return Some(NODE_22);
           									}
                    										if (*img_row11.add((c + 2) as usize)).to_bool() {
                    											return Some(NODE_20);
                    										}
                              											if (*img_row11.add((c) as usize)).to_bool() {
                              												*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
                              												return Some(cl_tree_4);
                              											}
                                         												*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
                                         												return Some(cl_tree_3);
           								}
                   									return Some(NODE_2);
    						}
          							if (*img_row01.add((c - 1) as usize)).to_bool() {
          								*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
          								return Some(cl_tree_7);
          							}
                 								if (*img_row00.add((c - 1) as usize)).to_bool() {
                 									*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
                 									return Some(cl_tree_7);
                 								}
                         									*img_labels_row00.add(c as usize) = solver.new_label();
                         									return Some(cl_tree_7);
    					}
         						return Some(NODE_1);
}
cl_tree_12 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(cl_break_0_8); }return Some(cl_break_1_12); }
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row01.add((c - 1) as usize)).to_bool() {
						if (*img_row11.add((c + 1) as usize)).to_bool() {
							if (*img_row11.add((c - 2) as usize)).to_bool() {
								return Some(NODE_40);
							}
       								*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
       								return Some(cl_tree_11);
						}
      							if (*img_row00.add((c + 1) as usize)).to_bool() {
      								return Some(NODE_32);
      							}
             								if (*img_row11.add((c) as usize)).to_bool() {
             									if (*img_row11.add((c - 2) as usize)).to_bool() {
             										return Some(NODE_39);
             									}
                      										*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
                      										return Some(cl_tree_10);
             								}
                     									*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
                     									return Some(cl_tree_9);
					}
     						return Some(NODE_14);
				}
    					if (*img_row01.add((c) as usize)).to_bool() {
    						if (*img_row00.add((c + 1) as usize)).to_bool() {
    							if (*img_row01.add((c - 1) as usize)).to_bool() {
    								if (*img_row11.add((c + 1) as usize)).to_bool() {
    									if (*img_row11.add((c - 2) as usize)).to_bool() {
    										return Some(NODE_38);
    									}
             										*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
             										return Some(cl_tree_6);
    								}
            									return Some(NODE_32);
    							}
           								return Some(NODE_2);
    						}
          							return Some(NODE_17);
    					}
         						return Some(NODE_1);
}
		NODE_41=> {
		if (*img_row11.add((c - 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c - 2) as usize), solver);
		}
		else {
		return Some(NODE_42);
		}
				}
		NODE_43=> {
		if (*img_row01.add((c) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
		}
		else {
			*img_labels_row00.add(c as usize) = 0.elem();
		}
				}
		NODE_42=> {
		if (*img_row11.add((c) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
		}
		else {
			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
		}
				}
		NODE_44=> {
		if (*img_row11.add((c) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
		}
		else {
			if (*img_row11.add((c - 1) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c - 2) as usize);
			}
			else {
				*img_labels_row00.add(c as usize) = solver.new_label();
			}
		}
				}
		NODE_45=> {
		if (*img_row01.add((c) as usize)).to_bool() {
			if (*img_row01.add((c - 1) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
			}
			else {
				*img_labels_row00.add(c as usize) = solver.new_label();
			}
		}
		else {
			*img_labels_row00.add(c as usize) = 0.elem();
		}
				}
		NODE_46=> {
		if (*img_row12.add((c - 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
		}
		else {
			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
		}
				}
		NODE_47=> {
		if (*img_row11.add((c) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
		}
		else {
			*img_labels_row00.add(c as usize) = solver.new_label();
		}
				}
		NODE_48=> {
		if (*img_row01.add((c) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = solver.new_label();
		}
		else {
			*img_labels_row00.add(c as usize) = 0.elem();
		}
				}
cl_break_0_0 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_47);
				}
    					return Some(NODE_48);
		return None;}
cl_break_0_1 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_44);
				}
    					return Some(NODE_48);
		return None;}
cl_break_0_2 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_41);
				}
    					return Some(NODE_43);
		return None;}
cl_break_0_3 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
				}
				else {
					return Some(NODE_43);
				}
		return None;}
cl_break_0_4 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row01.add((c - 1) as usize)).to_bool() {
						return Some(NODE_41);
					}
     						return Some(NODE_44);
				}
    					return Some(NODE_45);
		return None;}
cl_break_0_5 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row01.add((c - 1) as usize)).to_bool() {
						return Some(NODE_42);
					}
     						return Some(NODE_47);
				}
    					return Some(NODE_45);
		return None;}
cl_break_0_6 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row01.add((c - 1) as usize)).to_bool() {
						if (*img_row11.add((c) as usize)).to_bool() {
							return Some(NODE_46);
						}
      							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
					}
					else {
						return Some(NODE_47);
					}
				}
				else {
					return Some(NODE_45);
				}
		return None;}
cl_break_0_7 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row00.add((c - 1) as usize)).to_bool() {
						*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
					}
					else {
						if (*img_row01.add((c - 1) as usize)).to_bool() {
							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
						}
						else {
							if (*img_row11.add((c) as usize)).to_bool() {
								*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
							}
							else {
								*img_labels_row00.add(c as usize) = *img_labels_row12.add((c - 2) as usize);
							}
						}
					}
				}
				else {
					if (*img_row01.add((c) as usize)).to_bool() {
						if (*img_row01.add((c - 1) as usize)).to_bool() {
							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
						}
						else {
							if (*img_row00.add((c - 1) as usize)).to_bool() {
								*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
							}
							else {
								*img_labels_row00.add(c as usize) = solver.new_label();
							}
						}
					}
					else {
						*img_labels_row00.add(c as usize) = 0.elem();
					}
				}
		return None;}
cl_break_0_8 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row01.add((c - 1) as usize)).to_bool() {
						if (*img_row11.add((c) as usize)).to_bool() {
							if (*img_row11.add((c - 2) as usize)).to_bool() {
								return Some(NODE_46);
							}
       								*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
						}
						else {
							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
						}
					}
					else {
						return Some(NODE_47);
					}
				}
				else {
					return Some(NODE_45);
				}
		return None;}
		NODE_49=> {
		if (*img_row12.add((c - 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
		}
		else {
			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
		}
				}
		NODE_50=> {
		if (*img_row11.add((c + 1) as usize)).to_bool() {
			if (*img_row12.add((c) as usize)).to_bool() {
			return Some(NODE_49);
			}
   				*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
		}
		else {
			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
		}
				}
		NODE_51=> {
		if (*img_row11.add((c + 1) as usize)).to_bool() {
		return Some(NODE_52);
		}
  			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
				}
		NODE_52=> {
		if (*img_row11.add((c) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
		}
		else {
			if (*img_row12.add((c) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
			}
			else {
				*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
			}
		}
				}
		NODE_53=> {
		if (*img_row01.add((c) as usize)).to_bool() {
			if (*img_row00.add((c + 1) as usize)).to_bool() {
			return Some(NODE_54);
			}
   			return Some(NODE_55);
		}
  		return Some(NODE_56);
				}
		NODE_55=> {
		if (*img_row01.add((c - 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
		}
		else {
			*img_labels_row00.add(c as usize) = solver.new_label();
		}
				}
		NODE_54=> {
		if (*img_row01.add((c - 1) as usize)).to_bool() {
		return Some(NODE_57);
		}
  		return Some(NODE_58);
				}
		NODE_59=> {
		if (*img_row00.add((c + 1) as usize)).to_bool() {
			if (*img_row11.add((c + 1) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
			}
			else {
				*img_labels_row00.add(c as usize) = solver.new_label();
			}
		}
		else {
		return Some(NODE_60);
		}
				}
		NODE_61=> {
		if (*img_row11.add((c + 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
		}
		else {
			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
		}
				}
		NODE_62=> {
		if (*img_row01.add((c - 1) as usize)).to_bool() {
			if (*img_row11.add((c + 1) as usize)).to_bool() {
			return Some(NODE_63);
			}
   				if (*img_row11.add((c) as usize)).to_bool() {
   				return Some(NODE_49);
   				}
       					*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
		}
		else {
		return Some(NODE_58);
		}
				}
		NODE_63=> {
		if (*img_row12.add((c - 1) as usize)).to_bool() {
		return Some(NODE_52);
		}
  			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
				}
		NODE_64=> {
		if (*img_row11.add((c + 1) as usize)).to_bool() {
			if (*img_row12.add((c) as usize)).to_bool() {
			return Some(NODE_65);
			}
   				*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
		}
		else {
			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
		}
				}
		NODE_65=> {
		if (*img_row11.add((c - 2) as usize)).to_bool() {
		return Some(NODE_49);
		}
  			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
				}
		NODE_66=> {
		if (*img_row01.add((c - 1) as usize)).to_bool() {
			if (*img_row11.add((c + 1) as usize)).to_bool() {
				if (*img_row11.add((c - 2) as usize)).to_bool() {
				return Some(NODE_63);
				}
    					*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
			}
			else {
				if (*img_row11.add((c) as usize)).to_bool() {
				return Some(NODE_65);
				}
    					*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
			}
		}
		else {
		return Some(NODE_58);
		}
				}
		NODE_67=> {
		if (*img_row01.add((c) as usize)).to_bool() {
			if (*img_row00.add((c + 1) as usize)).to_bool() {
			return Some(NODE_58);
			}
   				*img_labels_row00.add(c as usize) = solver.new_label();
		}
		else {
		return Some(NODE_56);
		}
				}
		NODE_56=> {
		if (*img_row00.add((c + 1) as usize)).to_bool() {
		return Some(NODE_58);
		}
  		return Some(NODE_60);
				}
		NODE_58=> {
		if (*img_row11.add((c + 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
		}
		else {
			if (*img_row11.add((c) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
			}
			else {
				*img_labels_row00.add(c as usize) = solver.new_label();
			}
		}
				}
		NODE_68=> {
		if (*img_row11.add((c + 1) as usize)).to_bool() {
			if (*img_row11.add((c) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
			}
			else {
				if (*img_row11.add((c - 1) as usize)).to_bool() {
					if (*img_row12.add((c) as usize)).to_bool() {
						*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
					}
					else {
						*img_labels_row00.add(c as usize) = LabelsSolver::merge(LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver), *img_labels_row12.add((c - 2) as usize), solver);
					}
				}
				else {
					*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
				}
			}
		}
		else {
			if (*img_row11.add((c - 1) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c - 2) as usize), solver);
			}
			else {
			return Some(NODE_69);
			}
		}
				}
		NODE_70=> {
		if (*img_row11.add((c + 1) as usize)).to_bool() {
			if (*img_row11.add((c) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
			}
			else {
				if (*img_row11.add((c - 1) as usize)).to_bool() {
				return Some(NODE_71);
				}
    					*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
			}
		}
		else {
			if (*img_row11.add((c) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
			}
			else {
				if (*img_row11.add((c - 1) as usize)).to_bool() {
					*img_labels_row00.add(c as usize) = *img_labels_row12.add((c - 2) as usize);
				}
				else {
					*img_labels_row00.add(c as usize) = solver.new_label();
				}
			}
		}
				}
		NODE_57=> {
		if (*img_row11.add((c + 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
		}
		else {
		return Some(NODE_69);
		}
				}
		NODE_60=> {
		if (*img_row01.add((c + 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = solver.new_label();
		}
		else {
			*img_labels_row00.add(c as usize) = 0.elem();
		}
				}
		NODE_71=> {
		if (*img_row12.add((c) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
		}
		else {
			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row12.add((c) as usize), *img_labels_row12.add((c - 2) as usize), solver);
		}
				}
		NODE_69=> {
		if (*img_row11.add((c) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
		}
		else {
			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
		}
				}
cl_break_1_0 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_58);
				}
    					return Some(NODE_67);
		return None;}
cl_break_1_1 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_70);
				}
    					return Some(NODE_67);
		return None;}
cl_break_1_2 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_68);
				}
    					if (*img_row01.add((c) as usize)).to_bool() {
    						if (*img_row00.add((c + 1) as usize)).to_bool() {
    							return Some(NODE_57);
    						}
          							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
    					}
    					else {
    						return Some(NODE_56);
    					}
		return None;}
cl_break_1_3 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_61);
				}
    					if (*img_row01.add((c) as usize)).to_bool() {
    						if (*img_row00.add((c + 1) as usize)).to_bool() {
    							return Some(NODE_61);
    						}
          							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
    					}
    					else {
    						return Some(NODE_59);
    					}
		return None;}
cl_break_1_4 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_50);
				}
    					if (*img_row01.add((c) as usize)).to_bool() {
    						if (*img_row00.add((c + 1) as usize)).to_bool() {
    							return Some(NODE_50);
    						}
          							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
    					}
    					else {
    						return Some(NODE_59);
    					}
		return None;}
cl_break_1_5 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
				}
				else {
					if (*img_row01.add((c) as usize)).to_bool() {
						*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
					}
					else {
						if (*img_row00.add((c + 1) as usize)).to_bool() {
							*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
						}
						else {
							return Some(NODE_60);
						}
					}
				}
		return None;}
cl_break_1_6 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_51);
				}
    					if (*img_row01.add((c) as usize)).to_bool() {
    						if (*img_row00.add((c + 1) as usize)).to_bool() {
    							return Some(NODE_51);
    						}
          							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
    					}
    					else {
    						return Some(NODE_56);
    					}
		return None;}
cl_break_1_7 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row01.add((c - 1) as usize)).to_bool() {
						return Some(NODE_68);
					}
     						return Some(NODE_70);
				}
    					return Some(NODE_53);
		return None;}
cl_break_1_8 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_64);
				}
    					if (*img_row01.add((c) as usize)).to_bool() {
    						if (*img_row00.add((c + 1) as usize)).to_bool() {
    							return Some(NODE_64);
    						}
          							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
    					}
    					else {
    						return Some(NODE_59);
    					}
		return None;}
cl_break_1_9 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_54);
				}
    					return Some(NODE_53);
		return None;}
cl_break_1_10 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_62);
				}
    					if (*img_row01.add((c) as usize)).to_bool() {
    						if (*img_row00.add((c + 1) as usize)).to_bool() {
    							return Some(NODE_62);
    						}
          							return Some(NODE_55);
    					}
         						return Some(NODE_56);
		return None;}
cl_break_1_11 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row00.add((c - 1) as usize)).to_bool() {
						return Some(NODE_51);
					}
     						if (*img_row01.add((c - 1) as usize)).to_bool() {
     							return Some(NODE_51);
     						}
           							if (*img_row11.add((c + 1) as usize)).to_bool() {
           								if (*img_row11.add((c) as usize)).to_bool() {
           									*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
           								}
           								else {
           									return Some(NODE_71);
           								}
           							}
           							else {
           								if (*img_row11.add((c) as usize)).to_bool() {
           									*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
           								}
           								else {
           									*img_labels_row00.add(c as usize) = *img_labels_row12.add((c - 2) as usize);
           								}
           							}
				}
				else {
					if (*img_row01.add((c) as usize)).to_bool() {
						if (*img_row00.add((c + 1) as usize)).to_bool() {
							if (*img_row00.add((c - 1) as usize)).to_bool() {
								return Some(NODE_51);
							}
       								if (*img_row01.add((c - 1) as usize)).to_bool() {
       									return Some(NODE_51);
       								}
               									return Some(NODE_58);
						}
      							if (*img_row01.add((c - 1) as usize)).to_bool() {
      								*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
      							}
      							else {
      								if (*img_row00.add((c - 1) as usize)).to_bool() {
      									*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
      								}
      								else {
      									*img_labels_row00.add(c as usize) = solver.new_label();
      								}
      							}
					}
					else {
						return Some(NODE_56);
					}
				}
		return None;}
cl_break_1_12 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_66);
				}
    					if (*img_row01.add((c) as usize)).to_bool() {
    						if (*img_row00.add((c + 1) as usize)).to_bool() {
    							return Some(NODE_66);
    						}
          							return Some(NODE_55);
    					}
         						return Some(NODE_56);
		return None;}
    } None})(label)
{
label = next;
}
}}
