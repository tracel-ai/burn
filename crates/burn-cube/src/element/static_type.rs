use crate::CubeType;

/// Types that exist within a cube function
/// but should not be turned into a JIT variable
/// For instance: a bool that determines at compile time
/// if we should unroll a for loop or not

impl CubeType for bool {
    type ExpandType = bool;
}

impl CubeType for u32 {
    type ExpandType = u32;
}

impl CubeType for f32 {
    type ExpandType = f32;
}

impl CubeType for i32 {
    type ExpandType = i32;
}
