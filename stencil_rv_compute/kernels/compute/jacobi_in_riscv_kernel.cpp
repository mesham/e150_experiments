// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    uint32_t l1_uk_buffer_addr = get_compile_time_arg_val(0);
    uint32_t l1_ukp1_buffer_addr = get_compile_time_arg_val(1);

    uint32_t x_size  = get_compile_time_arg_val(2);
    uint32_t y_size  = get_compile_time_arg_val(3);
    uint32_t num_its = get_compile_time_arg_val(4);

    uint32_t in_cb_addr = get_compile_time_arg_val(5);
    uint32_t out_cb_addr = get_compile_time_arg_val(6);

    std::uint32_t data_size=x_size*y_size*4;

    constexpr auto cb_in0 = tt::CB::c_in0;
    
    cb_wait_front(cb_in0, 1);

    float* uk_data=(float*) l1_uk_buffer_addr;
    float* ukp1_data=(float*) l1_ukp1_buffer_addr;

    for (uint32_t n=0;n<x_size*y_size;n++) {
        uk_data[n]=((float*) in_cb_addr)[n];
        ukp1_data[n]=uk_data[n];
    }

    cb_pop_front(cb_in0, 1);

    for (uint32_t k=0;k<num_its;k++) {
        for (uint32_t i=1;i<x_size-1;i++) {
            for (uint32_t j=1;j<y_size-1;j++) {
                // shifting data here, as the compute cores don't like doing FP arithmetic (the compiler
                // configured memory map overruns)
                ukp1_data[(i*y_size)+j]=uk_data[((i+1)*y_size)+j];
            }
        }
        float * tmp=ukp1_data;
        ukp1_data=uk_data;
        uk_data=tmp;
    }

    constexpr auto cb_out0 = tt::CB::c_out0;

    for (uint32_t n=0;n<x_size*y_size;n++) {
        ((float*) out_cb_addr)[n]=uk_data[n];
    }
    cb_push_back(cb_out0, 1);
}
}
