@group(0)
@binding(0)
var<storage, read_write> lhs: array<elem>;

@group(0)
@binding(1)
var<storage, read> rhs: array<elem>;

@group(0)
@binding(2)
var<storage, read> info: array<u32>;

@compute
@workgroup_size(WORKGROUP_SIZE_X, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim: u32 = info[0];
    var index_rhs: u32 = 0u;

    for (var i: u32 = 0u; i < dim; i++) {
        let stride_lhs = info[i + 1u];
        let stride_rhs = info[i + dim + 1u];
        let shape_rhs = info[i + 3u * dim + 1u];

        index_rhs += global_id.x / stride_lhs % shape_rhs * stride_rhs;
    }

    BODY
}
