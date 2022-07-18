use crate::node::{Ones, Zeros};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::RwLock;

#[derive(new, Clone, Copy)]
pub struct Partial<T> {
    pub parent_position: usize,
    pub partial: T,
}

pub trait Node<T>: Send + Sync {
    fn parent_left(&self) -> Option<Partial<T>>;
    fn parent_right(&self) -> Option<Partial<T>>;
    fn position(&self) -> usize;
}

#[derive(new)]
pub struct RootNode {
    index: usize,
}

impl<T: Clone> Node<T> for RootNode {
    fn parent_left(&self) -> Option<Partial<T>> {
        None
    }
    fn parent_right(&self) -> Option<Partial<T>> {
        None
    }

    fn position(&self) -> usize {
        self.index
    }
}

#[derive(new)]
pub struct BinaryOperationNode<T> {
    left: Partial<T>,
    right: Partial<T>,
    index: usize,
}

impl<T: Clone + Send + Sync> Node<T> for BinaryOperationNode<T> {
    fn parent_left(&self) -> Option<Partial<T>> {
        Some(self.left.clone())
    }
    fn parent_right(&self) -> Option<Partial<T>> {
        Some(self.right.clone())
    }
    fn position(&self) -> usize {
        self.index
    }
}

#[derive(new)]
pub struct UnaryOperationNode<T> {
    left: Partial<T>,
    index: usize,
}

impl<T: Clone + Send + Sync> Node<T> for UnaryOperationNode<T> {
    fn parent_left(&self) -> Option<Partial<T>> {
        Some(self.left.clone())
    }

    fn parent_right(&self) -> Option<Partial<T>> {
        None
    }

    fn position(&self) -> usize {
        self.index
    }
}

/// Tape holding the computation graph
pub struct Tape<T> {
    pub nodes: RwLock<Vec<Box<dyn Node<T>>>>,
    pub grads: RwLock<Vec<T>>,
}

impl<T> Tape<T>
where
    T: Zeros<T> + Ones<T> + Clone + Send + Sync + 'static,
{
    pub fn new() -> Tape<T> {
        Tape {
            nodes: RwLock::new(Vec::new()),
            grads: RwLock::new(Vec::new()),
        }
    }

    pub fn register(&self, value: T) -> Var<T> {
        let mut nodes = self.nodes.write().unwrap();
        let mut grads = self.grads.write().unwrap();

        let position = nodes.len();

        nodes.push(Box::new(RootNode::new(position)));
        grads.push(value.zeros());

        Var::new(self, position, value)
    }

    pub fn register_unary(&self, partial: T, index: usize, value: T) -> Var<T> {
        let mut nodes = self.nodes.write().unwrap();
        let mut grads = self.grads.write().unwrap();

        let len = nodes.len();

        nodes.push(Box::new(UnaryOperationNode::new(
            Partial::new(index, partial),
            len,
        )));
        grads.push(value.zeros());

        Var::new(self, len, value)
    }

    pub fn register_binary(
        &self,
        partial_left: T,
        partial_right: T,
        parent_left_index: usize,
        parent_right_index: usize,
        value: T,
    ) -> Var<T> {
        let mut nodes = self.nodes.write().unwrap();
        let mut grads = self.grads.write().unwrap();

        let len = nodes.len();

        nodes.push(Box::new(BinaryOperationNode::new(
            Partial::new(parent_left_index, partial_left),
            Partial::new(parent_right_index, partial_right),
            len,
        )));
        grads.push(value.zeros());

        Var::new(self, len, value)
    }
}

/// Variable for computations
#[derive(new, Clone, Copy)]
pub struct Var<'t, T> {
    /// Pointer to the tape holding the corresponding node
    pub tape: &'t Tape<T>,
    /// Index of the node in the tape
    pub index: usize,
    /// Value
    pub v: T,
}

impl<T> Var<'_, T>
where
    T: Mul<Output = T> + Clone + Zeros<T> + Ones<T> + Add<Output = T> + Copy,
{
    /// Perform back propagation
    pub fn backprop(self) -> Grad<T> {
        // vector storing the gradients
        let nodes = self.tape.nodes.read().unwrap();
        let tape_len = nodes.len();
        let mut grad: Vec<T> = self.tape.grads.read().unwrap().to_owned();
        grad[self.index] = grad[self.index].ones();

        // iterate through the tape from back to front
        // because during forward pass, we always store new nodes at the end
        // of the tape, when we do the backward pass we can
        // just incrementally add partial * adjoint
        for i in (0..tape_len).rev() {
            let node = &nodes[i];
            // increment gradient contribution to the left parent
            if let Some(parent) = node.parent_left() {
                let grad_value = grad[parent.parent_position];
                grad[parent.parent_position] = grad_value + (parent.partial * grad[i].clone());
            }

            // increment gradient contribution to the right parent
            if let Some(parent) = node.parent_right() {
                let grad_value = grad[parent.parent_position];
                grad[parent.parent_position] = grad_value + (parent.partial * grad[i].clone());
            }
        }

        // TODO: reset tape;
        Grad { grad }
    }
}

impl<'t, T> Add for Var<'t, T>
where
    T: Ones<T> + Zeros<T> + Add<Output = T> + Clone + Send + Sync + 'static,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.tape.register_binary(
            self.v.ones(),
            self.v.ones(),
            self.index,
            rhs.index,
            self.v + rhs.v,
        )
    }
}

impl<'t, T> Sub for Var<'t, T>
where
    T: Ones<T> + Zeros<T> + Neg<Output = T> + Sub<Output = T> + Clone + Send + Sync + 'static,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.tape.register_binary(
            self.v.ones(),
            -self.v.ones(),
            self.index,
            rhs.index,
            self.v - rhs.v,
        )
    }
}

impl<'t, T> Neg for Var<'t, T>
where
    T: Ones<T> + Zeros<T> + Neg<Output = T> + Sub<Output = T> + Clone + Send + Sync + 'static,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.tape
            .register_unary(-self.v.ones(), self.index, -self.v)
    }
}

impl<'t, T> Mul for Var<'t, T>
where
    T: Ones<T> + Zeros<T> + Mul<Output = T> + Clone + Send + Sync + 'static,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.tape.register_binary(
            rhs.v.clone(),
            self.v.clone(),
            self.index,
            rhs.index,
            self.v * rhs.v,
        )
    }
}
impl<'t, T> Div for Var<'t, T>
where
    T: Ones<T>
        + Zeros<T>
        + Neg<Output = T>
        + Div<Output = T>
        + Mul<Output = T>
        + Clone
        + Send
        + Sync
        + 'static,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.tape.register_binary(
            self.v.ones() / rhs.v.clone(),
            -self.v.clone() / (rhs.v.clone() * rhs.v.clone()),
            self.index,
            rhs.index,
            self.v.clone() / rhs.v.clone(),
        )
    }
}

impl<'t> Mul<Var<'t, f64>> for f64 {
    type Output = Var<'t, f64>;

    fn mul(self, rhs: Var<'t, f64>) -> Self::Output {
        rhs.tape.register_unary(self, rhs.index, self * rhs.v)
    }
}

/// Struct holding gradients
#[derive(Debug)]
pub struct Grad<T> {
    pub grad: Vec<T>,
}

impl<T> Grad<T>
where
    T: Clone,
{
    /// Get the gradient with respect to a variable
    pub fn wrt(&self, var: Var<T>) -> T {
        self.grad[var.index].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::*;

    #[test]
    fn addition_test() {
        let tape = Tape::new();
        let x = tape.register(1.0);
        let y = tape.register(4.0);
        let z = x + y;
        let grad = z.backprop();
        assert!(approx_eq!(f64, grad.wrt(x), 1.0, ulps = 5));
        assert!(approx_eq!(f64, grad.wrt(y), 1.0, ulps = 5));
    }

    #[test]
    fn mul_test() {
        let tape = Tape::new();
        let x = tape.register(1.0);
        let y = tape.register(4.0);
        let z = x * y;
        let grad = z.backprop();
        assert!(approx_eq!(f64, grad.wrt(x), y.v, ulps = 5));
        assert!(approx_eq!(f64, grad.wrt(y), x.v, ulps = 5));
    }

    #[test]
    fn neg_test() {
        let tape = Tape::new();
        let x = tape.register(1.0);
        let z = -x;
        let grad = z.backprop();
        assert!(approx_eq!(f64, grad.wrt(x), -1.0, ulps = 5));
    }

    #[test]
    fn multiple_multiplications_test() {
        let tape = Tape::new();
        let x = tape.register(1.0);
        let y = tape.register(1.0);
        let z = -2.0 * x + x * x * x * y;
        let grad = z.backprop();
        assert!(approx_eq!(f64, grad.wrt(x), 1.0, ulps = 5));
    }
}
