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
}

pub trait Node<Out>: std::fmt::Debug {
    fn id(&self) -> NodeId;
    fn grad(&mut self) -> Out;
    fn value(&self) -> Out;
    fn update_grad(&mut self, grad: Out);
}
pub type NodeRef<T> = Rc<RefCell<dyn Node<T>>>;

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

impl<Out> Node<Out> for RootNode<Out>
where
    Out: Zeros<Out> + Clone + Mul<Output = Out> + Add<Output = Out>,
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
        let node = BinaryOpsNode::new($lhs, $rhs, $out);
        std::rc::Rc::new(std::cell::RefCell::new(node))
    }};
    ( input $input:expr, out $out:expr, ) => {{
        use $crate::graph::ops::SingleOpsNode;
        let node = SingleOpsNode::new($input, $out);
        std::rc::Rc::new(std::cell::RefCell::new(node))
    }};
    ( root $out:expr ) => {{
        use $crate::graph::node::RootNode;
        let node = RootNode::new($out);
        std::rc::Rc::new(std::cell::RefCell::new(node))
    }};
}
