pub(crate) fn checks_channels_div_groups(channels_in: usize, channels_out: usize, groups: usize) {
    let channels_in_div_by_group = channels_in % groups == 0;
    let channels_out_div_by_group = channels_out % groups == 0;

    if !channels_in_div_by_group || !channels_out_div_by_group {
        panic!(
            "Both channels must be divisible by the number of groups. Got \
             channels_in={channels_in}, channels_out={channels_out}, groups={groups}"
        );
    }
}
