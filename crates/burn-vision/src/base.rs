use derive_new::new;

/// 2D size used for vision ops.
#[derive(new, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Size {
    /// Width of the element
    pub width: usize,
    /// Height of the element
    pub height: usize,
}

/// 2D Point used for vision ops. Coordinates start at the top left.
#[derive(new, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Point {
    /// X (horizontal) coordinate
    pub x: usize,
    /// Y (vertical) coordinate
    pub y: usize,
}
