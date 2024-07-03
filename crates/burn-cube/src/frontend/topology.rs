//! In this file we use a trick where the constant has the same name as the module containing
//! the expand function, so that a user implicitly imports the expand function when importing the constant.

use crate::frontend::UInt;

macro_rules! constant {
    ($ident:ident, $var:expr, $doc:expr) => {
        #[doc = $doc]
        pub const $ident: UInt = UInt::new(0u32);

        #[allow(non_snake_case)]
        #[doc = $doc]
        pub mod $ident {
            use crate::frontend::{CubeContext, ExpandElement};

            /// Expansion of the constant variable.
            pub fn expand(_context: &mut CubeContext) -> ExpandElement {
                ExpandElement::Plain($var)
            }
        }
    };
}

constant!(
    SUBCUBE_DIM,
    crate::ir::Variable::SubcubeDim,
    r"
The total amount of working units in a subcube.
"
);

constant!(
    UNIT_POS,
    crate::ir::Variable::UnitPos,
    r"
The position of the working unit inside the cube, without regards to axis.
"
);

constant!(
    UNIT_POS_X,
    crate::ir::Variable::UnitPosX,
    r"
The position of the working unit inside the cube along the X axis.
"
);

constant!(
    UNIT_POS_Y,
    crate::ir::Variable::UnitPosY,
    r"
The position of the working unit inside the cube along the Y axis.
"
);

constant!(
    UNIT_POS_Z,
    crate::ir::Variable::UnitPosZ,
    r"
The position of the working unit inside the cube along the Z axis.
"
);

constant!(
    CUBE_DIM,
    crate::ir::Variable::CubeDim,
    r"
The total amount of working units in a cube.
"
);

constant!(
    CUBE_DIM_X,
    crate::ir::Variable::CubeDimX,
    r"
The dimension of the cube along the X axis.
"
);

constant!(
    CUBE_DIM_Y,
    crate::ir::Variable::CubeDimY,
    r"
The dimension of the cube along the Y axis.
"
);

constant!(
    CUBE_DIM_Z,
    crate::ir::Variable::CubeDimZ,
    r"
The dimension of the cube along the Z axis.
"
);

constant!(
    CUBE_POS,
    crate::ir::Variable::CubePos,
    r"
The cube position, without regards to axis.
"
);

constant!(
    CUBE_POS_X,
    crate::ir::Variable::CubePosX,
    r"
The cube position along the X axis.
"
);

constant!(
    CUBE_POS_Y,
    crate::ir::Variable::CubePosY,
    r"
The cube position along the Y axis.
"
);

constant!(
    CUBE_POS_Z,
    crate::ir::Variable::CubePosZ,
    r"
The cube position along the Z axis.
"
);
constant!(
    CUBE_COUNT,
    crate::ir::Variable::CubeCount,
    r"
The number of cubes launched.
"
);

constant!(
    CUBE_COUNT_X,
    crate::ir::Variable::CubeCountX,
    r"
The number of cubes launched along the X axis.
"
);

constant!(
    CUBE_COUNT_Y,
    crate::ir::Variable::CubeCountY,
    r"
The number of cubes launched along the Y axis.
"
);

constant!(
    CUBE_COUNT_Z,
    crate::ir::Variable::CubeCountZ,
    r"
The number of cubes launched along the Z axis.
"
);

constant!(
    ABSOLUTE_POS,
    crate::ir::Variable::AbsolutePos,
    r"
The position of the working unit in the whole cube kernel, without regards to cubes and axis.
"
);

constant!(
    ABSOLUTE_POS_X,
    crate::ir::Variable::AbsolutePosX,
    r"
The index of the working unit in the whole cube kernel along the X axis, without regards to cubes.
"
);

constant!(
    ABSOLUTE_POS_Y,
    crate::ir::Variable::AbsolutePosY,
    r"
The index of the working unit in the whole cube kernel along the Y axis, without regards to cubes.
"
);

constant!(
    ABSOLUTE_POS_Z,
    crate::ir::Variable::AbsolutePosZ,
    r"
The index of the working unit in the whole cube kernel along the Z axis, without regards to cubes.
"
);
