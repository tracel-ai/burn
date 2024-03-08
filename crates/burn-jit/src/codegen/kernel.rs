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
    execute_static_::<R, K, E, E, E>(inputs, outputs, scalar_elems, None, None, launch, client)
}

fn execute_static_<R, K, E1, E2, E3>(
    inputs: &[EagerHandle<R>],
    outputs: &[EagerHandle<R>],
    scalars_1: Option<&[E1]>,
    scalars_2: Option<&[E2]>,
    scalars_3: Option<&[E3]>,
    launch: WorkgroupLaunch,
    client: ComputeClient<R::Server, R::Channel>,
) where
    K: StaticKernelSource + 'static,
    R: Runtime,
    E1: JitElement,
    E2: JitElement,
    E3: JitElement,
{
    let settings = execute_settings(
        inputs, outputs, scalars_1, scalars_2, scalars_3, launch, &client,
    );
    let mut handles = settings.handles_tensors;
    let workgroup = settings.workgroup;

    handles.push(&settings.handle_info);
    for handle in settings.handles_scalars.iter() {
        handles.push(handle);
    }

    let kernel = Box::new(StaticKernel::<K>::new(workgroup));
    client.execute(kernel, &handles);
}

pub struct Execution<'h, K, R: Runtime, Scalars> {
    scalars: Scalars,
    client: ComputeClient<R::Server, R::Channel>,
    kernel: K,
    inputs: &'h [EagerHandle<'h, R>],
    outputs: &'h [EagerHandle<'h, R>],
}

impl<'h, K, R: Runtime> Execution<'h, K, R, ()> {
    pub fn start(
        kernel: K,
        client: ComputeClient<R::Server, R::Channel>,
    ) -> Execution<'h, K, R, ()> {
        Execution {
            scalars: (),
            client,
            kernel,
            inputs: &[],
            outputs: &[],
        }
    }

    #[allow(unused)]
    pub fn inputs(self, inputs: &'h [EagerHandle<'h, R>]) -> Execution<'h, K, R, ()> {
        Execution {
            scalars: self.scalars,
            client: self.client,
            kernel: self.kernel,
            inputs,
            outputs: self.outputs,
        }
    }

    pub fn outputs(self, outputs: &'h [EagerHandle<'h, R>]) -> Execution<'h, K, R, ()> {
        Execution {
            scalars: self.scalars,
            client: self.client,
            kernel: self.kernel,
            inputs: self.inputs,
            outputs,
        }
    }
}

impl<'h, K, R> Execution<'h, K, R, ()>
where
    K: DynamicKernelSource + 'static,
    R: Runtime,
{
    pub fn with_scalars<E>(self, scalars: &[E]) -> Execution<'h, K, R, (&[E],)> {
        Execution {
            scalars: (scalars,),
            client: self.client,
            kernel: self.kernel,
            inputs: self.inputs,
            outputs: self.outputs,
        }
    }
    /// Execute a dynamic kernel.
    #[allow(unused)]
    pub fn execute(self, launch: WorkgroupLaunch) {
        execute_dynamic_::<R, K, f32, f32, f32>(
            self.inputs,
            self.outputs,
            None,
            None,
            None,
            self.kernel,
            launch,
            self.client,
        )
    }
}

impl<'h, 'a, K, R, E> Execution<'h, K, R, (&'a [E],)>
where
    K: DynamicKernelSource + 'static,
    R: Runtime,
    E: JitElement,
{
    pub fn with_scalars<'b, E2>(
        self,
        scalars: &'b [E2],
    ) -> Execution<'h, K, R, (&'a [E], &'b [E2])> {
        Execution {
            scalars: (self.scalars.0, scalars),
            client: self.client,
            kernel: self.kernel,
            inputs: self.inputs,
            outputs: self.outputs,
        }
    }

    /// Execute a dynamic kernel.
    #[allow(unused)]
    pub fn execute(self, launch: WorkgroupLaunch) {
        execute_dynamic_::<R, K, E, f32, f32>(
            self.inputs,
            self.outputs,
            Some(self.scalars.0),
            None,
            None,
            self.kernel,
            launch,
            self.client,
        )
    }
}

impl<'h, 'a, 'b, K, R, E1, E2> Execution<'h, K, R, (&'a [E1], &'b [E2])>
where
    K: DynamicKernelSource + 'static,
    R: Runtime,
    E1: JitElement,
    E2: JitElement,
{
    #[allow(unused)]
    pub fn with_scalars<'c, E3>(
        self,
        scalars: &'c [E3],
    ) -> Execution<'h, K, R, (&'a [E1], &'b [E2], &'c [E3])> {
        Execution {
            scalars: (self.scalars.0, self.scalars.1, scalars),
            client: self.client,
            kernel: self.kernel,
            inputs: self.inputs,
            outputs: self.outputs,
        }
    }
    /// Execute a dynamic kernel.
    pub fn execute(self, launch: WorkgroupLaunch)
    where
        K: DynamicKernelSource + 'static,
        R: Runtime,
    {
        execute_dynamic_::<R, K, E1, E2, f32>(
            self.inputs,
            self.outputs,
            Some(self.scalars.0),
            Some(self.scalars.1),
            None,
            self.kernel,
            launch,
            self.client,
        )
    }
}

impl<'h, 'a, 'b, 'c, K, R, E1, E2, E3> Execution<'h, K, R, (&'a [E1], &'b [E2], &'c [E3])>
where
    K: DynamicKernelSource + 'static,
    R: Runtime,
    E1: JitElement,
    E2: JitElement,
    E3: JitElement,
{
    /// Execute a dynamic kernel.
    #[allow(unused)]
    pub fn execute(self, launch: WorkgroupLaunch) {
        execute_dynamic_::<R, K, E1, E2, E3>(
            self.inputs,
            self.outputs,
            Some(self.scalars.0),
            Some(self.scalars.1),
            Some(self.scalars.2),
            self.kernel,
            launch,
            self.client,
        )
    }
}

/// Execute a dynamic kernel.
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
    execute_dynamic_::<R, K, E, E, E>(
        inputs,
        outputs,
        scalar_elems,
        None,
        None,
        kernel,
        launch,
        client,
    )
}

fn execute_dynamic_<R, K, E1, E2, E3>(
    inputs: &[EagerHandle<R>],
    outputs: &[EagerHandle<R>],
    scalars_1: Option<&[E1]>,
    scalars_2: Option<&[E2]>,
    scalars_3: Option<&[E3]>,
    kernel: K,
    launch: WorkgroupLaunch,
    client: ComputeClient<R::Server, R::Channel>,
) where
    K: DynamicKernelSource + 'static,
    R: Runtime,
    E1: JitElement,
    E2: JitElement,
    E3: JitElement,
{
    let settings = execute_settings(
        inputs, outputs, scalars_1, scalars_2, scalars_3, launch, &client,
    );
    let mut handles = settings.handles_tensors;
    let workgroup = settings.workgroup;

    handles.push(&settings.handle_info);
    for handle in settings.handles_scalars.iter() {
        handles.push(handle);
    }

    let kernel: Box<dyn Kernel> = Box::new(DynamicKernel::new(kernel, workgroup));

    client.execute(kernel, &handles);
}

struct ExecuteSettings<'a, R: Runtime> {
    handles_tensors: Vec<&'a Handle<R::Server>>,
    handle_info: Handle<R::Server>,
    handles_scalars: Vec<Handle<R::Server>>,
    workgroup: WorkGroup,
}

fn execute_settings<'a, R: Runtime, E1: JitElement, E2: JitElement, E3: JitElement>(
    inputs: &'a [EagerHandle<R>],
    outputs: &'a [EagerHandle<R>],
    scalars_1: Option<&[E1]>,
    scalars_2: Option<&[E2]>,
    scalars_3: Option<&[E3]>,
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
    let mut handles_scalars = Vec::new();
    if let Some(values) = &scalars_1 {
        handles_scalars.push(client.create(bytemuck::cast_slice(values)));
    }
    if let Some(values) = &scalars_2 {
        handles_scalars.push(client.create(bytemuck::cast_slice(values)));
    }
    if let Some(values) = &scalars_3 {
        handles_scalars.push(client.create(bytemuck::cast_slice(values)));
    }

    let workgroup = match launch {
        WorkgroupLaunch::Custom(workgroup) => workgroup,
        _ => elemwise_workgroup(num_elems_output, WORKGROUP_DEFAULT),
    };

    ExecuteSettings {
        handles_tensors: handles,
        handle_info: info,
        handles_scalars,
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
