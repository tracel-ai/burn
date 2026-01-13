use burn_ir::TensorIr;

pub struct Node {
    inputs: Vec<TensorIr>,
}

pub struct Stream {
    nodes: Vec<Node>,
}
