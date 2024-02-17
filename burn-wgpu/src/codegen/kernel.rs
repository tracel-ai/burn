use crate::compute::StaticKernel;
use crate::element::JitElement;
use crate::kernel::{elemwise_workgroup, StaticKernelSource, WORKGROUP_DEFAULT};
use crate::Runtime;
use burn_compute::client::ComputeClient;

#[derive(new)]
pub struct StaticHandle<'a, R: Runtime> {
    handle: &'a burn_compute::server::Handle<R::Server>,
    strides: &'a [usize],
    shape: &'a [usize],
}

/// The position of the input or output to calculate the number of workgroups to launch.
pub enum WorkgroupLaunch {
    Input { pos: usize },
    Output { pos: usize },
}

/// Execute a static kernel.
///
///
/// The limitation from this method is that you can't launch a kernel with multiple types of
/// scalar.
pub fn execute_static<R: Runtime, K, E: JitElement>(
    inputs: &[StaticHandle<R>],
    outputs: &[StaticHandle<R>],
    scalar_elems: Option<&[E]>,
    launch: WorkgroupLaunch,
    client: ComputeClient<R::Server, R::Channel>,
) where
    K: StaticKernelSource + 'static,
{
    let mut info = Vec::new();
    let mut handles = Vec::with_capacity(inputs.len() + outputs.len() + 2);

    // Inner function to fill the info buffer.
    let mut register_info_tensor = |strides: &[usize], shape: &[usize]| {
        if info.is_empty() {
            info.push(strides.len() as u32);
        }

        for s in strides.iter() {
            info.push(*s as u32);
        }
        for s in shape.iter() {
            info.push(*s as u32);
        }
    };

    let mut num_elems_output = 0;

    // We start by registering the inputs.
    for (i, input) in inputs.iter().enumerate() {
        if let WorkgroupLaunch::Input { pos } = &launch {
            if i == *pos {
                num_elems_output = calculate_num_elems_dyn_rank(input.shape);
            }
        };
        register_info_tensor(input.strides, input.shape);
        handles.push(input.handle);
    }

    // Then we follow with the outputs.
    for (i, output) in outputs.iter().enumerate() {
        if let WorkgroupLaunch::Output { pos } = &launch {
            if i == *pos {
                num_elems_output = calculate_num_elems_dyn_rank(output.shape);
            }
        };
        register_info_tensor(output.strides, output.shape);
        handles.push(output.handle);
    }

    let info = &client.create(bytemuck::cast_slice(&info));
    handles.push(info);

    // Finally we finish with the named bindings.
    let mut scalars = None;
    if let Some(values) = &scalar_elems {
        scalars = Some(client.create(bytemuck::cast_slice(values)));
    }

    if let Some(scalars) = scalars.as_ref() {
        handles.push(scalars);
    }

    let workgroup = elemwise_workgroup(num_elems_output, WORKGROUP_DEFAULT);
    let kernel = Box::new(StaticKernel::<K>::new(workgroup));

    client.execute(kernel, &handles);
}

pub(crate) fn calculate_num_elems_dyn_rank(shape: &[usize]) -> usize {
    let mut num_elems = 1;
    for i in shape.iter() {
        num_elems *= i;
    }
    num_elems
}
