use crate::{ops::RecordedOpsRef, tape::Tape};
use std::{
    cell::RefCell,
    ops::{Add, Mul},
    rc::Rc,
};

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
        self.ops.record(tape)
    }
}
pub type NodeRef<Out> = Rc<Node<Out>>;

pub trait NodeState<Out>: std::fmt::Debug {
    fn id(&self) -> NodeId;
    fn grad(&mut self) -> Out;
    fn value(&self) -> Out;
    fn update_grad(&mut self, grad: Out);
}

pub type NodeStateRef<T> = Rc<RefCell<dyn NodeState<T>>>;

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

impl<Out> NodeState<Out> for RootNode<Out>
where
    Out: Zeros<Out> + Ones<Out> + Clone + Mul<Output = Out> + Add<Output = Out> + 'static,
    Out: std::fmt::Debug,
{
    fn id(&self) -> NodeId {
        self.id.clone()
    }
    fn grad(&mut self) -> Out {
        let grad_self: Out = match &self.grad {
            Some(val) => val.clone(),
            None => self.value.zeros(),
        };
        self.grad = Some(grad_self.clone());
        grad_self
    }

    fn value(&self) -> Out {
        self.value.clone()
    }

    fn update_grad(&mut self, grad: Out) {
        self.grad = Some(self.grad() + grad);
    }
}

#[macro_export]
macro_rules! node_init {
    ( lhs $lhs:expr, rhs $rhs:expr, out $out:expr, ) => {{
        use $crate::graph::ops::BinaryOpsNode;
        let node = BinaryOpsNode::new($lhs.state.clone(), $rhs.state.clone(), $out);
        std::rc::Rc::new(std::cell::RefCell::new(node))
    }};
    ( input $input:expr, out $out:expr, ) => {{
        use $crate::graph::ops::SingleOpsNode;
        let node = SingleOpsNode::new($input.state.clone(), $out);
        std::rc::Rc::new(std::cell::RefCell::new(node))
    }};
    ( root $out:expr ) => {{
        use $crate::graph::node::RootNode;
        let node = RootNode::new($out);
        std::rc::Rc::new(std::cell::RefCell::new(node))
    }};
}
