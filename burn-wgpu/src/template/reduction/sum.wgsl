fn sum_reduction(workgroup_id: u32) {
    var sum = elem(0);
    for (var i: u32 = 0u; i < WORKGROUP_SIZE_Xu; i++) {
        sum += data[i];
    }

    output[workgroup_id] = sum;
}
