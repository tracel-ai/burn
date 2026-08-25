no_analyze!{{
use lastLabels::{NODE_78, ll_tree_4, NODE_79, ll_tree_6, NODE_80, NODE_81, NODE_82, ll_tree_3, ll_tree_2, NODE_83, ll_tree_5, ll_tree_1, NODE_84, NODE_85, NODE_86, ll_tree_7, ll_tree_0, ll_break_0_0, ll_break_1_0, ll_break_0_1, ll_break_1_1, ll_break_0_2, ll_break_1_2, ll_break_1_3, ll_break_1_4, ll_break_1_5, ll_break_0_3, ll_break_1_6, ll_break_1_7, NODE_87, NODE_88, NODE_89, NODE_90, NODE_91, NODE_92, ll_};let mut label = entry;
while let Some(next) = (|label| -> Option<lastLabels> { match label {
		NODE_78=> {
		if (*img_row12.add((c - 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c + 2) as usize);
			return Some(ll_tree_4);
		}
  			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row12.add((c + 2) as usize), *img_labels_row12.add((c - 2) as usize), solver);
  			return Some(ll_tree_4);
				}
		NODE_79=> {
		if (*img_row12.add((c - 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
			return Some(ll_tree_6);
		}
  			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row12.add((c) as usize), *img_labels_row12.add((c - 2) as usize), solver);
  			return Some(ll_tree_6);
				}
		NODE_80=> {
		if (*img_row12.add((c) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
			return Some(ll_tree_6);
		}
  			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row12.add((c) as usize), *img_labels_row12.add((c - 2) as usize), solver);
  			return Some(ll_tree_6);
				}
		NODE_81=> {
		if (*img_row11.add((c + 2) as usize)).to_bool() {
			if (*img_row11.add((c) as usize)).to_bool() {
			return Some(NODE_82);
			}
   				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c + 2) as usize);
   				return Some(ll_tree_4);
		}
  			if (*img_row11.add((c) as usize)).to_bool() {
  				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
  				return Some(ll_tree_3);
  			}
     				*img_labels_row00.add(c as usize) = solver.new_label();
     				return Some(ll_tree_2);
				}
		NODE_83=> {
		if (*img_row00.add((c + 1) as usize)).to_bool() {
			if (*img_row11.add((c + 1) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
				return Some(ll_tree_5);
			}
   				if (*img_row11.add((c + 2) as usize)).to_bool() {
   					*img_labels_row00.add(c as usize) = *img_labels_row12.add((c + 2) as usize);
   					return Some(ll_tree_4);
   				}
       					*img_labels_row00.add(c as usize) = solver.new_label();
       					return Some(ll_tree_2);
		}
  			*img_labels_row00.add(c as usize) = 0.elem();
  			return Some(ll_tree_1);
				}
		NODE_84=> {
		if (*img_row00.add((c + 1) as usize)).to_bool() {
			if (*img_row11.add((c + 1) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
				return Some(ll_tree_5);
			}
   			return Some(NODE_81);
		}
  			*img_labels_row00.add(c as usize) = 0.elem();
  			return Some(ll_tree_1);
				}
		NODE_82=> {
		if (*img_row12.add((c + 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c + 2) as usize);
			return Some(ll_tree_4);
		}
  			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row12.add((c + 2) as usize), *img_labels_row12.add((c) as usize), solver);
  			return Some(ll_tree_4);
				}
		NODE_85=> {
		if (*img_row12.add((c + 1) as usize)).to_bool() {
			if (*img_row12.add((c) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c + 2) as usize);
				return Some(ll_tree_4);
			}
   				*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row12.add((c + 2) as usize), *img_labels_row12.add((c - 2) as usize), solver);
   				return Some(ll_tree_4);
		}
  			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row12.add((c + 2) as usize), *img_labels_row12.add((c - 2) as usize), solver);
  			return Some(ll_tree_4);
				}
		NODE_86=> {
		if (*img_row11.add((c + 1) as usize)).to_bool() {
			if (*img_row11.add((c) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
				return Some(ll_tree_6);
			}
   				if (*img_row12.add((c) as usize)).to_bool() {
   					*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
   					return Some(ll_tree_6);
   				}
       					*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
       					return Some(ll_tree_6);
		}
  			if (*img_row00.add((c + 1) as usize)).to_bool() {
  				if (*img_row11.add((c + 2) as usize)).to_bool() {
  					if (*img_row12.add((c + 1) as usize)).to_bool() {
  						if (*img_row11.add((c) as usize)).to_bool() {
  							*img_labels_row00.add(c as usize) = *img_labels_row12.add((c + 2) as usize);
  							return Some(ll_tree_4);
  						}
        							if (*img_row12.add((c) as usize)).to_bool() {
        								*img_labels_row00.add(c as usize) = *img_labels_row12.add((c + 2) as usize);
        								return Some(ll_tree_4);
        							}
               								*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
               								return Some(ll_tree_4);
  					}
       						*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
       						return Some(ll_tree_4);
  				}
      					*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
      					return Some(ll_tree_7);
  			}
     				*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
     				return Some(ll_tree_0);
				}
ll_tree_0 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(ll_break_0_0); }return Some(ll_break_1_0); }
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row11.add((c + 1) as usize)).to_bool() {
						*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
						return Some(ll_tree_6);
					}
     						if (*img_row00.add((c + 1) as usize)).to_bool() {
     							return Some(NODE_81);
     						}
           							if (*img_row11.add((c) as usize)).to_bool() {
           								*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
           								return Some(ll_tree_0);
           							}
                  								*img_labels_row00.add(c as usize) = solver.new_label();
                  								return Some(ll_tree_0);
				}
    					return Some(NODE_84);
}
ll_tree_1 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(ll_break_0_1); }return Some(ll_break_1_1); }
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row11.add((c + 1) as usize)).to_bool() {
						if (*img_row11.add((c) as usize)).to_bool() {
							*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
							return Some(ll_tree_6);
						}
      							if (*img_row11.add((c - 1) as usize)).to_bool() {
      								return Some(NODE_80);
      							}
             								*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
             								return Some(ll_tree_6);
					}
     						if (*img_row00.add((c + 1) as usize)).to_bool() {
     							if (*img_row11.add((c + 2) as usize)).to_bool() {
     								if (*img_row11.add((c) as usize)).to_bool() {
     									return Some(NODE_82);
     								}
             									if (*img_row11.add((c - 1) as usize)).to_bool() {
             										return Some(NODE_85);
             									}
                      										*img_labels_row00.add(c as usize) = *img_labels_row12.add((c + 2) as usize);
                      										return Some(ll_tree_4);
     							}
            								if (*img_row11.add((c) as usize)).to_bool() {
            									*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
            									return Some(ll_tree_3);
            								}
                    									if (*img_row11.add((c - 1) as usize)).to_bool() {
                    										*img_labels_row00.add(c as usize) = *img_labels_row12.add((c - 2) as usize);
                    										return Some(ll_tree_2);
                    									}
                             										*img_labels_row00.add(c as usize) = solver.new_label();
                             										return Some(ll_tree_2);
     						}
           							if (*img_row11.add((c) as usize)).to_bool() {
           								*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
           								return Some(ll_tree_0);
           							}
                  								if (*img_row11.add((c - 1) as usize)).to_bool() {
                  									*img_labels_row00.add(c as usize) = *img_labels_row12.add((c - 2) as usize);
                  									return Some(ll_tree_0);
                  								}
                          									*img_labels_row00.add(c as usize) = solver.new_label();
                          									return Some(ll_tree_0);
				}
    					return Some(NODE_84);
}
ll_tree_2 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(ll_break_0_2); }return Some(ll_break_1_2); }
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row11.add((c + 1) as usize)).to_bool() {
						*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
						return Some(ll_tree_6);
					}
     						if (*img_row00.add((c + 1) as usize)).to_bool() {
     							if (*img_row11.add((c + 2) as usize)).to_bool() {
     								*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
     								return Some(ll_tree_4);
     							}
            								*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
            								return Some(ll_tree_7);
     						}
           							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
           							return Some(ll_tree_0);
				}
    					return Some(NODE_83);
}
ll_tree_3 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(ll_break_0_2); }return Some(ll_break_1_3); }
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row11.add((c + 1) as usize)).to_bool() {
						if (*img_row12.add((c) as usize)).to_bool() {
							return Some(NODE_79);
						}
      							*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
      							return Some(ll_tree_6);
					}
     						if (*img_row00.add((c + 1) as usize)).to_bool() {
     							if (*img_row11.add((c + 2) as usize)).to_bool() {
     								if (*img_row12.add((c + 1) as usize)).to_bool() {
     									if (*img_row12.add((c) as usize)).to_bool() {
     										return Some(NODE_78);
     									}
              										*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
              										return Some(ll_tree_4);
     								}
             									*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
             									return Some(ll_tree_4);
     							}
            								*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
            								return Some(ll_tree_7);
     						}
           							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
           							return Some(ll_tree_0);
				}
    					return Some(NODE_83);
}
ll_tree_4 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(ll_break_0_2); }return Some(ll_break_1_4); }
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row11.add((c + 1) as usize)).to_bool() {
						*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
						return Some(ll_tree_6);
					}
     						if (*img_row00.add((c + 1) as usize)).to_bool() {
     							if (*img_row11.add((c + 2) as usize)).to_bool() {
     								if (*img_row12.add((c + 1) as usize)).to_bool() {
     									*img_labels_row00.add(c as usize) = *img_labels_row12.add((c + 2) as usize);
     									return Some(ll_tree_4);
     								}
             									*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
             									return Some(ll_tree_4);
     							}
            								*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
            								return Some(ll_tree_7);
     						}
           							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
           							return Some(ll_tree_0);
				}
    					if (*img_row00.add((c + 1) as usize)).to_bool() {
    						if (*img_row11.add((c + 1) as usize)).to_bool() {
    							*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
    							return Some(ll_tree_5);
    						}
          							if (*img_row11.add((c + 2) as usize)).to_bool() {
          								return Some(NODE_82);
          							}
                 								*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
                 								return Some(ll_tree_3);
    					}
         						*img_labels_row00.add(c as usize) = 0.elem();
         						return Some(ll_tree_1);
}
ll_tree_5 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(ll_break_0_2); }return Some(ll_break_1_5); }
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_86);
				}
    					return Some(NODE_84);
}
ll_tree_6 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(ll_break_0_3); }return Some(ll_break_1_6); }
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row00.add((c - 1) as usize)).to_bool() {
						return Some(NODE_86);
					}
     						if (*img_row11.add((c + 1) as usize)).to_bool() {
     							if (*img_row11.add((c) as usize)).to_bool() {
     								*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
     								return Some(ll_tree_6);
     							}
            								return Some(NODE_80);
     						}
           							if (*img_row00.add((c + 1) as usize)).to_bool() {
           								if (*img_row11.add((c + 2) as usize)).to_bool() {
           									if (*img_row11.add((c) as usize)).to_bool() {
           										return Some(NODE_82);
           									}
                    										return Some(NODE_85);
           								}
                   									if (*img_row11.add((c) as usize)).to_bool() {
                   										*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
                   										return Some(ll_tree_3);
                   									}
                            										*img_labels_row00.add(c as usize) = *img_labels_row12.add((c - 2) as usize);
                            										return Some(ll_tree_2);
           							}
                  								if (*img_row11.add((c) as usize)).to_bool() {
                  									*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
                  									return Some(ll_tree_0);
                  								}
                          									*img_labels_row00.add(c as usize) = *img_labels_row12.add((c - 2) as usize);
                          									return Some(ll_tree_0);
				}
    					return Some(NODE_84);
}
ll_tree_7 => {
if ({c+=2; c}) >= w - 2 { if c > w - 2 { return Some(ll_break_0_2); }return Some(ll_break_1_7); }
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row11.add((c + 1) as usize)).to_bool() {
						if (*img_row12.add((c) as usize)).to_bool() {
							if (*img_row11.add((c - 2) as usize)).to_bool() {
								return Some(NODE_79);
							}
       								*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
       								return Some(ll_tree_6);
						}
      							*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
      							return Some(ll_tree_6);
					}
     						if (*img_row00.add((c + 1) as usize)).to_bool() {
     							if (*img_row11.add((c + 2) as usize)).to_bool() {
     								if (*img_row12.add((c + 1) as usize)).to_bool() {
     									if (*img_row12.add((c) as usize)).to_bool() {
     										if (*img_row11.add((c - 2) as usize)).to_bool() {
     											return Some(NODE_78);
     										}
               											*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
               											return Some(ll_tree_4);
     									}
              										*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
              										return Some(ll_tree_4);
     								}
             									*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c + 2) as usize), solver);
             									return Some(ll_tree_4);
     							}
            								*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
            								return Some(ll_tree_7);
     						}
           							*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
           							return Some(ll_tree_0);
				}
    					return Some(NODE_83);
}
ll_break_0_0 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row11.add((c) as usize)).to_bool() {
						*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
					}
					else {
						*img_labels_row00.add(c as usize) = solver.new_label();
					}
				}
				else {
					*img_labels_row00.add(c as usize) = 0.elem();
				}
		return None;}
ll_break_0_1 => {
				if (*img_row00.add((c) as usize)).to_bool() {
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
				else {
					*img_labels_row00.add(c as usize) = 0.elem();
				}
		return None;}
ll_break_0_2 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
				}
				else {
					*img_labels_row00.add(c as usize) = 0.elem();
				}
		return None;}
ll_break_0_3 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row00.add((c - 1) as usize)).to_bool() {
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
				else {
					*img_labels_row00.add(c as usize) = 0.elem();
				}
		return None;}
		NODE_87=> {
		if (*img_row00.add((c + 1) as usize)).to_bool() {
		return Some(NODE_88);
		}
  			*img_labels_row00.add(c as usize) = 0.elem();
				}
		NODE_88=> {
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
		NODE_89=> {
		if (*img_row12.add((c - 1) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
		}
		else {
			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row12.add((c) as usize), *img_labels_row12.add((c - 2) as usize), solver);
		}
				}
		NODE_90=> {
		if (*img_row11.add((c + 1) as usize)).to_bool() {
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
		else {
			*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
		}
				}
		NODE_91=> {
		if (*img_row00.add((c + 1) as usize)).to_bool() {
			if (*img_row11.add((c + 1) as usize)).to_bool() {
				*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
			}
			else {
				*img_labels_row00.add(c as usize) = solver.new_label();
			}
		}
		else {
			*img_labels_row00.add(c as usize) = 0.elem();
		}
				}
		NODE_92=> {
		if (*img_row12.add((c) as usize)).to_bool() {
			*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
		}
		else {
			*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row12.add((c) as usize), *img_labels_row12.add((c - 2) as usize), solver);
		}
				}
ll_break_1_0 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_88);
				}
    					return Some(NODE_87);
		return None;}
ll_break_1_1 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row11.add((c + 1) as usize)).to_bool() {
						if (*img_row11.add((c) as usize)).to_bool() {
							*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
						}
						else {
							if (*img_row11.add((c - 1) as usize)).to_bool() {
								return Some(NODE_92);
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
				else {
					return Some(NODE_87);
				}
		return None;}
ll_break_1_2 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row11.add((c + 1) as usize)).to_bool() {
						*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
					}
					else {
						*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
					}
				}
				else {
					return Some(NODE_91);
				}
		return None;}
ll_break_1_3 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row11.add((c + 1) as usize)).to_bool() {
						if (*img_row12.add((c) as usize)).to_bool() {
							return Some(NODE_89);
						}
      							*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
					}
					else {
						*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
					}
				}
				else {
					return Some(NODE_91);
				}
		return None;}
ll_break_1_4 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
				}
				else {
					if (*img_row00.add((c + 1) as usize)).to_bool() {
						*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
					}
					else {
						*img_labels_row00.add(c as usize) = 0.elem();
					}
				}
		return None;}
ll_break_1_5 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					return Some(NODE_90);
				}
    					return Some(NODE_87);
		return None;}
ll_break_1_6 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row00.add((c - 1) as usize)).to_bool() {
						return Some(NODE_90);
					}
     						if (*img_row11.add((c + 1) as usize)).to_bool() {
     							if (*img_row11.add((c) as usize)).to_bool() {
     								*img_labels_row00.add(c as usize) = *img_labels_row12.add((c) as usize);
     							}
     							else {
     								return Some(NODE_92);
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
					return Some(NODE_87);
				}
		return None;}
ll_break_1_7 => {
				if (*img_row00.add((c) as usize)).to_bool() {
					if (*img_row11.add((c + 1) as usize)).to_bool() {
						if (*img_row12.add((c) as usize)).to_bool() {
							if (*img_row11.add((c - 2) as usize)).to_bool() {
								return Some(NODE_89);
							}
       								*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
						}
						else {
							*img_labels_row00.add(c as usize) = LabelsSolver::merge(*img_labels_row00.add((c - 2) as usize), *img_labels_row12.add((c) as usize), solver);
						}
					}
					else {
						*img_labels_row00.add(c as usize) = *img_labels_row00.add((c - 2) as usize);
					}
				}
				else {
					return Some(NODE_91);
				}
		return None;}
ll_ => {},
    } None})(label)
{
label = next;
}
}}
