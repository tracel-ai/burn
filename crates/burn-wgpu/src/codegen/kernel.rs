use crate::compute::{DynamicKernel, Kernel, StaticKernel, WorkGroup};
use crate::element::JitElement;
use crate::kernel::{
    elemwise_workgroup, DynamicKernelSource, StaticKernelSource, WORKGROUP_DEFAULT,
};
use crate::Runtime;
use burn_compute::client::ComputeClient;
use burn_compute::server::Handle;

#[derive(new)]
pub struct EagerHandle<'a, R: Runtime> {
    handle: &'a burn_compute::server::Handle<R::Server>,
    strides: &'a [usize],
    shape: &'a [usize],
}

/// The position of the input or output to calculate the number of workgroups to launch.
pub enum WorkgroupLaunch {
    Input { pos: usize },
    Output { pos: usize },
    Custom(WorkGroup),
}

/// Execute a static kernel.
///
///
/// The limitation from this method is that you can't launch a kernel with multiple types of
/// scalar.
pub fn execute_static<R, K, E>(
    inputs: &[EagerHandle<R>],
    outputs: &[EagerHandle<R>],
    scalar_elems: Option<&[E]>,
    launch: WorkgroupLaunch,
    client: ComputeClient<R::Server, R::Channel>,
) where
    K: StaticKernelSource + 'static,
    R: Runtime,
    E: JitElement,
{
    let settings = execute_settings(inputs, outputs, scalar_elems, launch, &client);
    let mut handles = settings.handles_tensors;
    let workgroup = settings.workgroup;

    handles.push(&settings.handle_info);
    if let Some(handle) = settings.handle_scalars.as_ref() {
        handles.push(handle);
    }

    let kernel = Box::new(StaticKernel::<K>::new(workgroup));
    client.execute(kernel, &handles);
}

/// Execute a dynamic kernel.
///
///
/// The limitation from this method is that you can't launch a kernel with multiple types of
/// scalar.
pub fn execute_dynamic<R, K, E>(
    inputs: &[EagerHandle<R>],
    outputs: &[EagerHandle<R>],
    scalar_elems: Option<&[E]>,
    kernel: K,
    launch: WorkgroupLaunch,
    client: ComputeClient<R::Server, R::Channel>,
) where
    K: DynamicKernelSource + 'static,
    R: Runtime,
    E: JitElement,
{
    let settings = execute_settings(inputs, outputs, scalar_elems, launch, &client);
    let mut handles = settings.handles_tensors;
    let workgroup = settings.workgroup;

    handles.push(&settings.handle_info);
    if let Some(handle) = settings.handle_scalars.as_ref() {
        handles.push(handle);
    }

    let kernel: Box<dyn Kernel> = Box::new(DynamicKernel::new(kernel, workgroup));

    client.execute(kernel, &handles);
}

struct ExecuteSettings<'a, R: Runtime> {
    handles_tensors: Vec<&'a Handle<R::Server>>,
    handle_info: Handle<R::Server>,
    handle_scalars: Option<Handle<R::Server>>,
    workgroup: WorkGroup,
}

fn execute_settings<'a, R: Runtime, E: JitElement>(
    inputs: &'a [EagerHandle<R>],
    outputs: &'a [EagerHandle<R>],
    scalar_elems: Option<&[E]>,
    launch: WorkgroupLaunch,
    client: &ComputeClient<R::Server, R::Channel>,
) -> ExecuteSettings<'a, R> {
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

    let info = client.create(bytemuck::cast_slice(&info));

    // Finally we finish with the named bindings.
    let mut scalars = None;
    if let Some(values) = &scalar_elems {
        scalars = Some(client.create(bytemuck::cast_slice(values)));
    }

    let workgroup = match launch {
        WorkgroupLaunch::Custom(workgroup) => workgroup,
        _ => elemwise_workgroup(num_elems_output, WORKGROUP_DEFAULT),
    };

    ExecuteSettings {
        handles_tensors: handles,
        handle_info: info,
        handle_scalars: scalars,
        workgroup,
    }
}

pub(crate) fn calculate_num_elems_dyn_rank(shape: &[usize]) -> usize {
    let mut num_elems = 1;
    for i in shape.iter() {
        num_elems *= i;
    }
    num_elems
}
