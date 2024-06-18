// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_its = get_compile_time_arg_val(0);

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_in1 = tt::CB::c_in1;
    constexpr auto cb_in2 = tt::CB::c_in2;
    constexpr auto cb_in3 = tt::CB::c_in3;
    constexpr auto cb_intermediate = tt::CB::c_intermed0;
    constexpr auto cb_out0 = tt::CB::c_out0;

    constexpr uint32_t dst_reg = 0;

    DPRINT << "Begin compute" << ENDL();

    acquire_dst(tt::DstMode::Half);

    binary_op_init_common(cb_in0, cb_in1, cb_intermediate);
    add_tiles_init(cb_in0, cb_in1);

    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);
    add_tiles(cb_in0, cb_in1, 0, 0, dst_reg);
    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);
    
    cb_reserve_back(cb_intermediate, 1);
    pack_tile(dst_reg, cb_intermediate);
    cb_push_back(cb_intermediate, 1);
    
    binary_op_init_common(cb_intermediate, cb_in2, cb_intermediate);
    add_tiles_init(cb_intermediate, cb_in2);

    cb_wait_front(cb_in2, 1);
    cb_wait_front(cb_intermediate, 1);
    add_tiles(cb_intermediate, cb_in2, 0, 0, dst_reg);
    cb_pop_front(cb_in2, 1);
    cb_pop_front(cb_intermediate, 1);    

    cb_reserve_back(cb_intermediate, 1);
    pack_tile(dst_reg, cb_intermediate);
    cb_push_back(cb_intermediate, 1);

    binary_op_init_common(cb_intermediate, cb_in3, cb_out0);
    add_tiles_init(cb_intermediate, cb_in3);

    cb_wait_front(cb_in3, 1);
    cb_wait_front(cb_intermediate, 1);
    add_tiles(cb_intermediate, cb_in3, 0, 0, dst_reg);
    cb_pop_front(cb_in3, 1);
    cb_pop_front(cb_intermediate, 1);

    cb_reserve_back(cb_out0, 1);
    pack_tile(dst_reg, cb_out0);
    cb_push_back(cb_out0, 1);

    release_dst(tt::DstMode::Half);

    DPRINT << "Compute ended" << ENDL();
}
}
