// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string.h>

void kernel_main() {
    std::uint32_t result_data_buffer_addr  = get_arg_val<uint32_t>(0);
    std::uint32_t result_data_buffer_noc_x = get_arg_val<uint32_t>(1);
    std::uint32_t result_data_buffer_noc_y = get_arg_val<uint32_t>(2);

    std::uint32_t x_size  = get_arg_val<uint32_t>(3);
    std::uint32_t y_size  = get_arg_val<uint32_t>(4);

    std::uint32_t data_size=x_size*y_size*4;

    uint64_t results_data_buffer_noc_addr = get_noc_addr(result_data_buffer_noc_x, result_data_buffer_noc_y, result_data_buffer_addr);

    constexpr uint32_t cb_id_out0 = tt::CB::c_out0;
    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

    cb_wait_front(cb_id_out0, 1);
    noc_async_write(l1_read_addr, results_data_buffer_noc_addr, data_size);
    noc_async_write_barrier();
    cb_pop_front(cb_id_out0, 1);
}
