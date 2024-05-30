//! In this file we use a trick where the constant has the same name as the module containing
//! the expand function, so that a user implicitly imports the expand function when importing the constant.

use crate::UInt;

macro_rules! constant {
    ($ident:ident, $var:expr, $doc:expr) => {
        #[doc = $doc]
        pub const $ident: UInt = UInt::new(0u32);

        #[allow(non_snake_case)]
        #[doc = $doc]
        pub mod $ident {
            use crate::{CubeContext, ExpandElement};

            /// Expansion of the constant variable.
            pub fn expand(_context: &mut CubeContext) -> ExpandElement {
                ExpandElement::Plain($var)
            }
        }
    };
}

constant!(
    UNIT_POS,
    crate::dialect::Variable::LocalInvocationIndex,
    r"
The position of the working unit inside the cube, without regards to axis.
"
);

constant!(
    UNIT_POS_X,
    crate::dialect::Variable::LocalInvocationIdX,
    r"
The position of the working unit inside the cube along the X axis.
"
);

constant!(
    UNIT_POS_Y,
    crate::dialect::Variable::LocalInvocationIdY,
    r"
The position of the working unit inside the cube along the Y axis.
"
);

constant!(
    UNIT_POS_Z,
    crate::dialect::Variable::LocalInvocationIdZ,
    r"
The position of the working unit inside the cube along the Z axis.
"
);

constant!(
    CUBE_DIM,
    crate::dialect::Variable::WorkgroupSize,
    r"
The total amount of working units in a cube.
"
);

constant!(
    CUBE_DIM_X,
    crate::dialect::Variable::WorkgroupSizeX,
    r"
The dimension of the cube along the X axis.
"
);

constant!(
    CUBE_DIM_Y,
    crate::dialect::Variable::WorkgroupSizeY,
    r"
The dimension of the cube along the Y axis.
"
);

constant!(
    CUBE_DIM_Z,
    crate::dialect::Variable::WorkgroupSizeZ,
    r"
The dimension of the cube along the Z axis.
"
);

constant!(
    CUBE_POS,
    crate::dialect::Variable::WorkgroupId,
    r"
The cube position, without regards to axis.
"
);

constant!(
    CUBE_POS_X,
    crate::dialect::Variable::WorkgroupIdX,
    r"
The cube position along the X axis.
"
);

constant!(
    CUBE_POS_Y,
    crate::dialect::Variable::WorkgroupIdY,
    r"
The cube position along the Y axis.
"
);

constant!(
    CUBE_POS_Z,
    crate::dialect::Variable::WorkgroupIdZ,
    r"
The cube position along the Z axis.
"
);
constant!(
    CUBE_COUNT,
    crate::dialect::Variable::NumWorkgroups,
    r"
The number of cubes launched.
"
);

constant!(
    CUBE_COUNT_X,
    crate::dialect::Variable::NumWorkgroupsX,
    r"
The number of cubes launched along the X axis.
"
);

constant!(
    CUBE_COUNT_Y,
    crate::dialect::Variable::NumWorkgroupsY,
    r"
The number of cubes launched along the Y axis.
"
);

constant!(
    CUBE_COUNT_Z,
    crate::dialect::Variable::NumWorkgroupsZ,
    r"
The number of cubes launched along the Z axis.
"
);

constant!(
    ABSOLUTE_POS,
    crate::dialect::Variable::Id,
    r"
The position of the working unit in the whole cube kernel, without regards to cubes and axis.
"
);

constant!(
    ABSOLUTE_POS_X,
    crate::dialect::Variable::GlobalInvocationIdX,
    r"
The index of the working unit in the whole cube kernel along the X axis, without regards to cubes.
"
);

constant!(
    ABSOLUTE_POS_Y,
    crate::dialect::Variable::GlobalInvocationIdY,
    r"
The index of the working unit in the whole cube kernel along the Y axis, without regards to cubes.
"
);

constant!(
    ABSOLUTE_POS_Z,
    crate::dialect::Variable::GlobalInvocationIdZ,
    r"
The index of the working unit in the whole cube kernel along the Z axis, without regards to cubes.
"
);
