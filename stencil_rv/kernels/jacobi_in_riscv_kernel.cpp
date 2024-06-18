// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string.h>

void kernel_main() {
    std::uint32_t domain_data_buffer_addr  = get_arg_val<uint32_t>(0);
    std::uint32_t domain_data_buffer_noc_x = get_arg_val<uint32_t>(1);
    std::uint32_t domain_data_buffer_noc_y = get_arg_val<uint32_t>(2);

    std::uint32_t results_data_buffer_addr  = get_arg_val<uint32_t>(3);
    std::uint32_t results_data_buffer_noc_x = get_arg_val<uint32_t>(4);
    std::uint32_t results_data_buffer_noc_y = get_arg_val<uint32_t>(5);

    std::uint32_t l1_uk_buffer_addr = get_arg_val<uint32_t>(6);
    std::uint32_t l1_ukp1_buffer_addr = get_arg_val<uint32_t>(7);

    std::uint32_t x_size  = get_arg_val<uint32_t>(8);
    std::uint32_t y_size  = get_arg_val<uint32_t>(9);
    std::uint32_t num_its = get_arg_val<uint32_t>(10);

    std::uint32_t data_size=x_size*y_size*4;

    uint64_t domain_data_buffer_noc_addr = get_noc_addr(domain_data_buffer_noc_x, domain_data_buffer_noc_y, domain_data_buffer_addr);
    uint64_t results_data_buffer_noc_addr = get_noc_addr(results_data_buffer_noc_x, results_data_buffer_noc_y, results_data_buffer_addr);

    noc_async_read(domain_data_buffer_noc_addr, l1_uk_buffer_addr, data_size);
    noc_async_read_barrier();

    float* uk_data=(float*) l1_uk_buffer_addr;
    float* ukp1_data=(float*) l1_ukp1_buffer_addr;

    memcpy(ukp1_data, uk_data, data_size);

    for (uint32_t k=0;k<num_its;k++) {
        for (uint32_t i=1;i<x_size-1;i++) {
            for (uint32_t j=1;j<y_size-1;j++) {
                ukp1_data[(i*y_size)+j]=0.25*(uk_data[((i+1)*y_size)+j] + uk_data[((i-1)*y_size)+j] + uk_data[(i*y_size)+(j+1)] + uk_data[(i*y_size)+(j-1)]);
            }
        }
        float * tmp=ukp1_data;
        ukp1_data=uk_data;
        uk_data=tmp;
    }

    noc_async_write(num_its % 2 == 0 ? l1_uk_buffer_addr : l1_ukp1_buffer_addr, results_data_buffer_noc_addr, data_size);
    noc_async_write_barrier();
}
