@group(0)
@binding(0)
var<storage, read> lhs: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> rhs: array<{{ elem }}>;

@group(0)
@binding(2)
var<storage, read_write> output: array<u32>;

@group(0)
@binding(3)
var<storage, read> info: array<u32>;


const WORKGROUP_SIZE_X = {{ workgroup_size_x }}u;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let id = global_id.y * (num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;
    let dim: u32 = info[0];
    var index_lhs: u32 = 0u;
    var index_rhs: u32 = 0u;

    for (var i: u32 = 1u; i <= dim; i++) {
        let stride_lhs = info[i];
        let stride_rhs = info[i + dim];
        let stride_output = info[i + 2u * dim];
        let shape_lhs = info[i + 3u * dim];
        let shape_rhs = info[i + 4u * dim];

        index_lhs += id / stride_output % shape_lhs * stride_lhs;
        index_rhs += id / stride_output % shape_rhs * stride_rhs;
    }

    {{ body }}
}
