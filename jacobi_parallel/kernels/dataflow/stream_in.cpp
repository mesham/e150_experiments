#include <cstdint>
#include <string.h>

// Byte alignment
#define ALIGNMENT 32
// Print debug stagements
#define PRINT_DEBUG 0

#define BATCH_SIZE_X 32
#define BATCH_SIZE_Y 32

std::uint32_t read_data(std::uint32_t address, std::uint32_t starting_address, std::uint32_t domain_data_buffer_noc_x, std::uint32_t domain_data_buffer_noc_y, std::uint32_t size, std::uint32_t read_in_buffer_addr) {
    std::uint32_t offset=(address - starting_address) % ALIGNMENT;
    std::uint32_t offset_start=address-offset;
    std::uint32_t read_size=size+offset;

    uint64_t domain_data_buffer_noc_addr = get_noc_addr(domain_data_buffer_noc_x, domain_data_buffer_noc_y, offset_start);
    noc_async_read(domain_data_buffer_noc_addr, read_in_buffer_addr, read_size);
    noc_async_read_barrier();
    return offset;
}

static inline uint16_t float_to_bfloat16(float val) {
    union {
        float f;
        uint32_t u;
    } ret;
    ret.f = val;
    return uint16_t(ret.u >> 16);
}

void kernel_main() {
    uint32_t cores_in_x = get_compile_time_arg_val(0);
    uint32_t cores_in_y = get_compile_time_arg_val(1);
    uint32_t rank_in_x = get_compile_time_arg_val(2);
    uint32_t rank_in_y = get_compile_time_arg_val(3);

    std::uint32_t area_one_data_buffer_addr  = get_arg_val<uint32_t>(0);
    std::uint32_t area_one_data_buffer_noc_x = get_arg_val<uint32_t>(1);
    std::uint32_t area_one_data_buffer_noc_y = get_arg_val<uint32_t>(2);

    std::uint32_t area_two_data_buffer_addr  = get_arg_val<uint32_t>(3);
    std::uint32_t area_two_data_buffer_noc_x = get_arg_val<uint32_t>(4);
    std::uint32_t area_two_data_buffer_noc_y = get_arg_val<uint32_t>(5);

    std::uint32_t read_in_buffer_addr = get_arg_val<uint32_t>(6);

    std::uint32_t x_size  = get_arg_val<uint32_t>(7);
    std::uint32_t y_size  = get_arg_val<uint32_t>(8);
    std::uint32_t num_its  = get_arg_val<uint32_t>(9);
    uint32_t semaphore_addr = get_arg_val<uint32_t>(10);

    std::uint32_t total_x_size = x_size + 2;
    std::uint32_t total_y_size = y_size + ((ALIGNMENT/2)*2);    // Number of elements is alignment div 2, and we have LHS and RHS

    std::uint32_t num_batches_x=x_size/BATCH_SIZE_X;
    std::uint32_t num_batches_y=y_size/BATCH_SIZE_Y;

    uint16_t* in_data=(uint16_t*) read_in_buffer_addr;

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_id_in1 = tt::CB::c_in1;
    constexpr uint32_t cb_id_in2 = tt::CB::c_in2;
    constexpr uint32_t cb_id_in3 = tt::CB::c_in3;
    constexpr auto cb_scalar = tt::CB::c_intermed1;

    volatile tt_l1_ptr uint32_t* semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);

    cb_reserve_back(cb_scalar, 1);
    uint16_t * scalar_buffer = (uint16_t*) get_write_ptr(cb_scalar);
    for (uint32_t i=0;i<1024;i++) {
        scalar_buffer[i]=float_to_bfloat16(0.25);
    }
    cb_push_back(cb_scalar, 1);

    std::uint32_t local_num_batches_y=num_batches_y/cores_in_y;
    uint32_t remainder_batch=num_batches_y-(local_num_batches_y * cores_in_y);
    std::uint32_t my_start_batch_y=(local_num_batches_y*rank_in_y) + remainder_batch;
    if (rank_in_y < remainder_batch) {
        my_start_batch_y=my_start_batch_y-(remainder_batch-rank_in_y);
        local_num_batches_y++;
    }
    std::uint32_t my_end_batch_y=my_start_batch_y+local_num_batches_y;

    std::uint32_t local_num_batches_x=num_batches_x/cores_in_x;
    remainder_batch=num_batches_x - (local_num_batches_x * cores_in_x);
    std::uint32_t my_start_batch_x=(local_num_batches_x*rank_in_x) + remainder_batch;
    if (rank_in_x < remainder_batch) {
        my_start_batch_x=my_start_batch_x-(remainder_batch-rank_in_x);
        local_num_batches_x++;
    }
    std::uint32_t my_end_batch_x=my_start_batch_x+local_num_batches_x;

#if PRINT_DEBUG==1
    DPRINT << "Core ("<<U32(rank_in_x)<<","<<U32(rank_in_y)<<") from X="<<U32(my_start_batch_x)<<","<<U32(my_end_batch_x)<<" Y="<<U32(my_start_batch_y)<<","<<U32(my_end_batch_y)<<ENDL();
#endif

    for (uint32_t it=0;it<num_its;it++){
        noc_semaphore_wait(semaphore_addr_ptr, it);
        for (uint32_t bx=my_start_batch_x;bx<my_end_batch_x;bx++) {
            std::uint32_t batch_x_start=bx*BATCH_SIZE_X*total_y_size;
            for (uint32_t by=my_start_batch_y;by<my_end_batch_y;by++) {
                std::uint32_t batch_y_start=by*BATCH_SIZE_Y;
#if PRINT_DEBUG==1                
                DPRINT << "Begin data load for iteration " << U32(it) << " batch x=" << U32(bx) << " y=" << U32(by) << ENDL();
#endif                

                // Grab the CBs for this specific tile
                cb_reserve_back(cb_id_in0, 1);
                cb_reserve_back(cb_id_in1, 1);
                cb_reserve_back(cb_id_in2, 1);
                cb_reserve_back(cb_id_in3, 1);

                uint16_t * yp1_buffer = (uint16_t*) get_write_ptr(cb_id_in0);
                uint16_t * ym1_buffer = (uint16_t*) get_write_ptr(cb_id_in1);
                uint16_t * xp1_buffer = (uint16_t*) get_write_ptr(cb_id_in2);
                uint16_t * xm1_buffer = (uint16_t*) get_write_ptr(cb_id_in3);

                for (uint32_t i=0;i<BATCH_SIZE_X+2;i++) {
                    std::uint32_t addr_offset=((batch_x_start+(i*total_y_size)+batch_y_start)*2) + (ALIGNMENT-2);

                    int start_idx;
                    if (it %2 == 0) {
                        start_idx=read_data(area_one_data_buffer_addr+addr_offset, area_one_data_buffer_addr, area_one_data_buffer_noc_x, area_one_data_buffer_noc_y, (BATCH_SIZE_Y+2)*2, read_in_buffer_addr);
                    } else {
                        start_idx=read_data(area_two_data_buffer_addr+addr_offset, area_two_data_buffer_addr, area_two_data_buffer_noc_x, area_two_data_buffer_noc_y, (BATCH_SIZE_Y+2)*2, read_in_buffer_addr);
                    }

                    if (i < BATCH_SIZE_X) {
                        memcpy(&xm1_buffer[(i*BATCH_SIZE_Y)], &in_data[1+(start_idx/2)], BATCH_SIZE_Y*2); // Starting from element 1 as that is the grid point
                    }
                    if (i >= 2) {
                        memcpy(&xp1_buffer[((i-2)*BATCH_SIZE_Y)], &in_data[1+(start_idx/2)], BATCH_SIZE_Y*2); // Starting from element 1 as that is the grid point
                    }
                    if (i >=1 && i < BATCH_SIZE_X+1) {
                        memcpy(&ym1_buffer[((i-1)*BATCH_SIZE_Y)], &in_data[start_idx/2], BATCH_SIZE_Y*2); // Starting from zeroth element, which is boundary condition
                        memcpy(&yp1_buffer[((i-1)*BATCH_SIZE_Y)], &in_data[2 + (start_idx/2)], BATCH_SIZE_Y*2); // Starting from second element, which is grid cell plus one
                    }
                }
                cb_push_back(cb_id_in0, 1);
                cb_push_back(cb_id_in1, 1);
                cb_push_back(cb_id_in2, 1);
                cb_push_back(cb_id_in3, 1);
#if PRINT_DEBUG==1                
                DPRINT << "Done data load for iteration " << U32(it) << " batch x=" << U32(bx) << " y=" << U32(by) << ENDL();
#endif                
            }
        }
    }
#if PRINT_DEBUG==1    
    DPRINT << "Leaving stream in" << ENDL();
#endif    
}
