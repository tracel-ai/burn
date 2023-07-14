@group(0)
@binding(0)
var<storage, read> grad: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> indices: array<{{ int }}>;

@group(0)
@binding(2)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(3)
var<storage, read> info: array<u32, 16>;

const WORKGROUP_SIZE_X = {{ workgroup_size_x }}u;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let id = global_id.y * (num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;

    let grad_stride_0 = info[0];
    let grad_stride_1 = info[1];
    let grad_stride_2 = info[2];
    let grad_stride_3 = info[3];
    let grad_shape_0 = info[4];
    let grad_shape_1 = info[5];
    let grad_shape_2 = info[6];
    let grad_shape_3 = info[7];

    let output_stride_0 = info[8];
    let output_stride_1 = info[9];
    let output_stride_2 = info[10];
    let output_stride_3 = info[11];
    let output_shape_0 = info[12];
    let output_shape_1 = info[13];
    let output_shape_2 = info[14];
    let output_shape_3 = info[15];

    let b = id / grad_stride_0 % grad_shape_0;
    let c = id / grad_stride_1 % grad_shape_1;

    let index = u32(indices[id]);
    let index_h = index / output_shape_3;
    let index_w = index % output_shape_3;
    let index_output = b * output_stride_0 + c * output_stride_1 + index_h * output_stride_2 + index_w * output_stride_3;

    var num_elems = 1u;
    num_elems *= grad_shape_0;
    num_elems *= grad_shape_1;
    num_elems *= grad_shape_2;
    num_elems *= grad_shape_3;

    if id < num_elems {
        output[index_output] += grad[id];
    }
}
