use crate::{ops::RecordedOpsRef, state::NodeStateImpl, tape::Tape};
use std::{cell::RefCell, ops::Add, rc::Rc};

#[derive(PartialEq, Eq, Debug, Clone, Hash)]
pub struct NodeId {
    value: String,
}

impl NodeId {
    pub fn new() -> Self {
        Self {
            value: nanoid::nanoid!(),
        }
    }
    pub fn to_string(&self) -> String {
        self.value.to_string()
    }
}

#[derive(new, Debug)]
pub struct Node<Out> {
    pub state: NodeStateRef<Out>,
    pub ops: RecordedOpsRef,
}

impl<Out> Node<Out> {
    pub fn record(&self, tape: &mut Tape) {
        let mut all_ops = self.ops.parents_ops();
        tape.add(self.ops.clone());

        loop {
            if all_ops.len() == 0 {
                return;
            }
            let ops = all_ops.pop().unwrap();
            all_ops.append(&mut ops.parents_ops());
            tape.add(ops);
        }
    }
}
pub type NodeRef<Out> = Rc<Node<Out>>;
pub type NodeStateRef<T> = Rc<RefCell<NodeStateImpl<T>>>;

pub trait Zeros<T> {
    fn zeros(&self) -> T;
}
pub trait Ones<T> {
    fn ones(&self) -> T;
}

#[derive(Debug)]
pub struct RootNode<Out> {
    pub id: NodeId,
    pub value: Out,
    pub grad: Option<Out>,
}

impl<Out> RootNode<Out> {
    pub fn new(value: Out) -> Self {
        Self {
            id: NodeId::new(),
            value,
            grad: None,
        }
    }
}

#[macro_export]
macro_rules! node_init {
    ( lhs $lhs:expr, rhs $rhs:expr, out $out:expr, ) => {{
        use $crate::graph::state::NodeStateImpl;
        let node = NodeStateImpl::new($out);
        std::rc::Rc::new(std::cell::RefCell::new(node))
    }};
    ( input $input:expr, out $out:expr, ) => {{
        use $crate::graph::state::NodeStateImpl;
        let node = NodeStateImpl::new($out);
        std::rc::Rc::new(std::cell::RefCell::new(node))
    }};
    ( root $out:expr ) => {{
        use $crate::graph::state::NodeStateImpl;
        let node = NodeStateImpl::new($out);
        std::rc::Rc::new(std::cell::RefCell::new(node))
    }};
}
