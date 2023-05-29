@group(0)
@binding(0)
var<storage, read_write> lhs: array<FLOAT>;

@group(0)
@binding(1)
var<storage, read_write> rhs: FLOAT;

@group(0)
@binding(2)
var<storage, read_write> output: array<FLOAT>;

@compute
@workgroup_size(WORKGROUP_SIZE_X, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    output[global_id.x] = lhs[global_id.x] + rhs;
}
