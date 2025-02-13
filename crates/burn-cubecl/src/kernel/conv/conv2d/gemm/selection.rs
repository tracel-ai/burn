use super::{
    algorithm::{Algorithm, ImplicitCmmaConv},
    precision::ConvPrecision,
};
use crate::CubeRuntime;
use cubecl::linalg::matmul::components::{CompleteStageTiling, MatmulSelection, MatmulSize};

pub struct ConvSelection {
    pub matmul: MatmulSelection,
}

pub trait ConvSelector<A: Algorithm> {
    fn select_kernel<R: CubeRuntime, CS: ConvPrecision>(plane_dim: u32)
        -> (A::Selection, A::Input);
}

/// Large m stage size for the usual case where `batch_size * out_h * out_w` is significantly larger
/// than `out_channels`
pub struct Large;
/// Balanced stage size for cases where `batch_size * out_h * out_w` is relatively small and `k` or
/// `out_channels` is relatively large
pub struct Balanced;

impl ConvSelector<ImplicitCmmaConv> for Large {
    fn select_kernel<R: CubeRuntime, CS: ConvPrecision>(
        plane_dim: u32,
    ) -> (
        <ImplicitCmmaConv as Algorithm>::Selection,
        <ImplicitCmmaConv as Algorithm>::Input,
    ) {
        let selection = MatmulSelection {
            tile_shape: MatmulSize {
                m: 16,
                n: 16,
                k: 16,
            },
            tile_count: MatmulSize { m: 8, n: 4, k: 2 },
            plane_dim,
        };
        let config_input = CompleteStageTiling {
            tile_shape: selection.tile_shape,
            tile_count: selection.tile_count,
        };

        let selection = ConvSelection { matmul: selection };

        (selection, config_input)
    }
}

impl ConvSelector<ImplicitCmmaConv> for Balanced {
    fn select_kernel<R: CubeRuntime, CS: ConvPrecision>(
        plane_dim: u32,
    ) -> (
        <ImplicitCmmaConv as Algorithm>::Selection,
        <ImplicitCmmaConv as Algorithm>::Input,
    ) {
        let selection = MatmulSelection {
            tile_shape: MatmulSize {
                m: 16,
                n: 16,
                k: 16,
            },
            tile_count: MatmulSize { m: 4, n: 2, k: 4 },
            plane_dim,
        };
        let config_input = CompleteStageTiling {
            tile_shape: selection.tile_shape,
            tile_count: selection.tile_count,
        };

        let selection = ConvSelection { matmul: selection };

        (selection, config_input)
    }
}
