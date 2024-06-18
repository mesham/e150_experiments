// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string.h>

void kernel_main() {
    std::uint32_t domain_data_buffer_addr  = get_arg_val<uint32_t>(0);
    std::uint32_t domain_data_buffer_noc_x = get_arg_val<uint32_t>(1);
    std::uint32_t domain_data_buffer_noc_y = get_arg_val<uint32_t>(2);

    std::uint32_t x_size  = get_arg_val<uint32_t>(3);
    std::uint32_t y_size  = get_arg_val<uint32_t>(4);

    std::uint32_t data_size=x_size*y_size*4;

    uint64_t domain_data_buffer_noc_addr = get_noc_addr(domain_data_buffer_noc_x, domain_data_buffer_noc_y, domain_data_buffer_addr);

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);

    cb_reserve_back(cb_id_in0, 1);
    noc_async_read(domain_data_buffer_noc_addr, l1_write_addr_in0, data_size);
    noc_async_read_barrier();
    cb_push_back(cb_id_in0, 1);
}
