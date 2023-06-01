@group(0)
@binding(0)
var<storage, read> lhs: array<elem>;

@group(0)
@binding(1)
var<storage, read> rhs: elem;

@group(0)
@binding(2)
var<storage, read_write> output: array<elem>;

@compute
@workgroup_size(WORKGROUP_SIZE_X, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    BODY
}
