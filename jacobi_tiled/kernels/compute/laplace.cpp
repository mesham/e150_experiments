// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api.h"
#include "debug/dprint.h"

// Print debug stagements
#define PRINT_DEBUG 0

#define BATCH_SIZE_X 32
#define BATCH_SIZE_Y 32

namespace NAMESPACE {
void MAIN {
    uint32_t x_size = get_compile_time_arg_val(0);
    uint32_t y_size = get_compile_time_arg_val(1);
    uint32_t num_its = get_compile_time_arg_val(2);

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_in1 = tt::CB::c_in1;
    constexpr auto cb_in2 = tt::CB::c_in2;
    constexpr auto cb_in3 = tt::CB::c_in3;
    constexpr auto cb_intermediate = tt::CB::c_intermed0;
    constexpr auto cb_scalar = tt::CB::c_intermed1;
    constexpr auto cb_out = tt::CB::c_out0;

    constexpr uint32_t dst0 = 0;

    std::uint32_t num_batches_x=x_size/BATCH_SIZE_X;
    std::uint32_t num_batches_y=y_size/BATCH_SIZE_Y;
    
    cb_wait_front(cb_scalar, 1);

    init_sfpu(tt::CB::c_in0);
    binary_op_init_common(cb_in0, cb_in1, cb_intermediate);
    for (uint32_t it=0;it<num_its;it++) {
#if PRINT_DEBUG==1
        DPRINT << "Begin compute iteration " << U32(it) << ENDL();
#endif

        for (uint32_t bx=0;bx<num_batches_x;bx++) {
            for (uint32_t by=0;by<num_batches_y;by++) {
                acquire_dst(tt::DstMode::Full);

                // CBs arguments are ignored
                add_tiles_init(cb_in0, cb_in1);
                cb_wait_front(cb_in0, 1);
                cb_wait_front(cb_in1, 1);
                add_tiles(cb_in0, cb_in1, 0, 0, dst0);
                cb_pop_front(cb_in1, 1);
                cb_pop_front(cb_in0, 1);

                cb_reserve_back(cb_intermediate, 1);
                pack_tile(dst0, cb_intermediate);
                cb_push_back(cb_intermediate, 1);
                release_dst(tt::DstMode::Full);

                // We have to release DST and reaquire between these, and reinit
                acquire_dst(tt::DstMode::Full);
                add_tiles_init(cb_in2, cb_intermediate);
                cb_wait_front(cb_in2, 1);
                cb_wait_front(cb_intermediate, 1);
                add_tiles(cb_in2, cb_intermediate, 0, 0, dst0);
                cb_pop_front(cb_intermediate, 1);
                cb_pop_front(cb_in2, 1);

                cb_reserve_back(cb_intermediate, 1);
                pack_tile(dst0, cb_intermediate);
                cb_push_back(cb_intermediate, 1);
                release_dst(tt::DstMode::Full);

                acquire_dst(tt::DstMode::Full);
                add_tiles_init(cb_in3, cb_intermediate);
                cb_wait_front(cb_intermediate, 1);
                cb_wait_front(cb_in3, 1);
                add_tiles(cb_in3, cb_intermediate, 0, 0, dst0);
                cb_pop_front(cb_in3, 1);
                cb_pop_front(cb_intermediate, 1);

                cb_reserve_back(cb_intermediate, 1);
                pack_tile(dst0, cb_intermediate);
                cb_push_back(cb_intermediate, 1);
                release_dst(tt::DstMode::Full);

                acquire_dst(tt::DstMode::Full);
                mul_tiles_init(cb_in0, cb_in1);
                cb_wait_front(cb_intermediate, 1);
                mul_tiles(cb_scalar, cb_intermediate, 0, 0, dst0);
                cb_pop_front(cb_intermediate, 1);

                cb_reserve_back(cb_out, 1);
                pack_tile(dst0, cb_out);
                cb_push_back(cb_out, 1);

                release_dst(tt::DstMode::Full);
            }
        }
#if PRINT_DEBUG==1
        DPRINT << "Compute iteration ended" << ENDL();
#endif
    }
    cb_pop_front(cb_scalar, 1);
#if PRINT_DEBUG==1
    DPRINT << "Leaving compute" << ENDL();
#endif
}
}
