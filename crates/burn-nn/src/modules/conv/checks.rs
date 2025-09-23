pub(crate) fn checks_channels_div_groups(channels_in: usize, channels_out: usize, groups: usize) {
    let channels_in_div_by_group = channels_in.is_multiple_of(groups);
    let channels_out_div_by_group = channels_out.is_multiple_of(groups);

    if !channels_in_div_by_group || !channels_out_div_by_group {
        panic!(
            "Both channels must be divisible by the number of groups. Got \
             channels_in={channels_in}, channels_out={channels_out}, groups={groups}"
        );
    }
}

// https://github.com/tracel-ai/burn/issues/2676
/// Only symmetric padding is currently supported. As such, using `Same` padding with an even kernel
/// size is not supported as it will not produce the same output size.
pub(crate) fn check_same_padding_support(kernel_size: &[usize]) {
    for k in kernel_size.iter() {
        if k % 2 == 0 {
            unimplemented!("Same padding with an even kernel size is not supported");
        }
    }
}
