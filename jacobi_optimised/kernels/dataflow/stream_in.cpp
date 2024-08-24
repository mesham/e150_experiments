#include <cstdint>
#include <string.h>

// Byte alignment
#define ALIGNMENT 32
// Print debug stagements
#define PRINT_DEBUG 0
// Synchronise between iterations
#define SYNC_ITS 0

#define BATCH_SIZE_X 32
#define BATCH_SIZE_Y 32

std::uint32_t read_data(std::uint32_t address, std::uint32_t size, std::uint32_t read_in_buffer_addr, char blocking, InterleavedAddrGen<true> addr_gen, std::uint32_t page_size) {
    std::uint32_t offset=address % ALIGNMENT;
    std::uint32_t offset_start=address-offset;
    std::uint32_t page_id=offset_start / page_size;
    std::uint32_t page_offset=offset_start % page_size;
    std::uint32_t read_size=size+offset;

    uint64_t domain_data_buffer_noc_addr = addr_gen.get_noc_addr(page_id, page_offset);
    noc_async_read(domain_data_buffer_noc_addr, read_in_buffer_addr, read_size);
    if (blocking) noc_async_read_barrier();
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

uint32_t load_batch(uint32_t batch_x_start, uint32_t total_y_size, uint32_t batch_y_start, uint32_t buffer_bytes_per_line, uint32_t read_in_buffer_addr_base, InterleavedAddrGen<true> addr_gen, std::uint32_t page_size) {
    uint32_t start_idx; // Extracting this out works as it's the same offset for all the retrieves
    for (uint32_t i=0;i<BATCH_SIZE_X+2;i++) {
        std::uint32_t addr_offset=((batch_x_start+(i*total_y_size)+batch_y_start)*2) + (ALIGNMENT-2);
        uint32_t buffer_addr_offset=i*buffer_bytes_per_line;

        start_idx=read_data(addr_offset, (BATCH_SIZE_Y+2)*2, read_in_buffer_addr_base+buffer_addr_offset, 0, addr_gen, page_size);
    }
    return start_idx;
}

uint32_t do_load(uint32_t batch_x_start, uint32_t total_y_size, uint32_t batch_y_start, uint32_t buffer_bytes_per_line, uint32_t read_in_buffer_addr_base, 
                uint32_t it, uint32_t db_idx, uint32_t buffer_bytes_per_block, InterleavedAddrGen<true> addr_gen_one, InterleavedAddrGen<true> addr_gen_two, std::uint32_t page_size) {
    uint32_t read_in_buffer_addr=read_in_buffer_addr_base+(db_idx == 1 ? buffer_bytes_per_block: 0);
    if (it %2 == 0) {
        return load_batch(batch_x_start, total_y_size, batch_y_start, buffer_bytes_per_line, read_in_buffer_addr, addr_gen_one, page_size);
    } else {
        return load_batch(batch_x_start, total_y_size, batch_y_start, buffer_bytes_per_line, read_in_buffer_addr, addr_gen_two, page_size);
    }
}


void kernel_main() {
    std::uint32_t area_one_data_buffer_addr  = get_arg_val<uint32_t>(0);
    std::uint32_t area_two_data_buffer_addr  = get_arg_val<uint32_t>(1);
    std::uint32_t read_in_buffer_addr = get_arg_val<uint32_t>(2);

    std::uint32_t x_size  = get_arg_val<uint32_t>(3);
    std::uint32_t y_size  = get_arg_val<uint32_t>(4);
    std::uint32_t num_its  = get_arg_val<uint32_t>(5);
    uint32_t page_size = get_arg_val<uint32_t>(6);
    uint32_t cores_in_x = get_arg_val<uint32_t>(7);
    uint32_t cores_in_y = get_arg_val<uint32_t>(8);
    uint32_t semaphore_addr = get_arg_val<uint32_t>(9);

    uint32_t rank_in_x = my_x[0]-1;
    uint32_t rank_in_y = my_y[0]-1;

    std::uint32_t total_x_size = x_size + 2;
    std::uint32_t total_y_size = y_size + ((ALIGNMENT/2)*2);    // Number of elements is alignment div 2, and we have LHS and RHS

    InterleavedAddrGen<true> addr_gen_one{.bank_base_address = area_one_data_buffer_addr, .page_size = page_size};
    InterleavedAddrGen<true> addr_gen_two{.bank_base_address = area_two_data_buffer_addr, .page_size = page_size};

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
                
    std::uint32_t buffer_bytes_per_line=(BATCH_SIZE_Y+2+(ALIGNMENT/2))*2;
    uint32_t e=buffer_bytes_per_line%ALIGNMENT;
    if (e > 0) buffer_bytes_per_line+=(ALIGNMENT-e);

    std::uint32_t buffer_bytes_per_block=buffer_bytes_per_line*(BATCH_SIZE_X+2);

#if PRINT_DEBUG==1
    DPRINT << "Core ("<<U32(rank_in_x)<<","<<U32(rank_in_y)<<") from X="<<U32(my_start_batch_x)<<","<<U32(my_end_batch_x)<<" Y="<<U32(my_start_batch_y)<<","<<U32(my_end_batch_y)<<ENDL();
#endif

    uint32_t start_idxs[2];
    for (uint32_t it=0;it<num_its;it++){
#if SYNC_ITS == 1    
        noc_semaphore_wait(semaphore_addr_ptr, it);
#endif        
        uint32_t db_idx=0;
        for (uint32_t bx=my_start_batch_x;bx<my_end_batch_x;bx++) {
            std::uint32_t batch_x_start=bx*BATCH_SIZE_X*total_y_size;
            for (uint32_t by=my_start_batch_y;by<my_end_batch_y;by++) {
                std::uint32_t batch_y_start=by*BATCH_SIZE_Y;
#if PRINT_DEBUG==1                
                DPRINT << "Begin data load for iteration " << U32(it) << " batch x=" << U32(bx) << " y=" << U32(by) << ENDL();
#endif  
                // Complete all reads on noc
                noc_async_read_barrier();
                if (db_idx == 2) db_idx=0;

                if (bx != my_end_batch_x-1 && by != my_end_batch_y-1) {
                    // Issue next read apart from the last iteration through
                    start_idxs[db_idx]=do_load(batch_x_start, total_y_size, batch_y_start, buffer_bytes_per_line, read_in_buffer_addr, it, 
                                                db_idx, buffer_bytes_per_block, addr_gen_one, addr_gen_two, page_size);
                    db_idx++;
                }

                if (bx == my_start_batch_x && by == my_start_batch_y) {
                    // Special case, for first batch we need to wait for it and then issue next batch data load
                    noc_async_read_barrier();
                    start_idxs[db_idx]=do_load(batch_x_start, total_y_size, batch_y_start, buffer_bytes_per_line, read_in_buffer_addr, it, 
                                                db_idx, buffer_bytes_per_block, addr_gen_one, addr_gen_two, page_size);
                    db_idx++;
                }

                // Grab the CBs for this specific tile
                cb_reserve_back(cb_id_in0, 1);
                cb_reserve_back(cb_id_in1, 1);
                cb_reserve_back(cb_id_in2, 1);
                cb_reserve_back(cb_id_in3, 1);

                uint16_t * yp1_buffer = (uint16_t*) get_write_ptr(cb_id_in0);
                uint16_t * ym1_buffer = (uint16_t*) get_write_ptr(cb_id_in1);
                uint16_t * xp1_buffer = (uint16_t*) get_write_ptr(cb_id_in2);
                uint16_t * xm1_buffer = (uint16_t*) get_write_ptr(cb_id_in3);

                uint32_t start_idx=start_idxs[db_idx == 1 ? 1 : 0];
                for (uint32_t i=0;i<BATCH_SIZE_X+2;i++) {
                    uint32_t buffer_addr_offset=((i*buffer_bytes_per_line)+(db_idx == 1 ? buffer_bytes_per_block: 0))/2;

                    if (i < BATCH_SIZE_X) {
                        memcpy(&xm1_buffer[(i*BATCH_SIZE_Y)], &in_data[buffer_addr_offset+1+(start_idx/2)], BATCH_SIZE_Y*2); // Starting from element 1 as that is the grid point
                    }
                    if (i >= 2) {
                        memcpy(&xp1_buffer[((i-2)*BATCH_SIZE_Y)], &in_data[buffer_addr_offset+1+(start_idx/2)], BATCH_SIZE_Y*2); // Starting from element 1 as that is the grid point
                    }
                    if (i >=1 && i < BATCH_SIZE_X+1) {
                        //DPRINT << BF16(in_data[buffer_addr_offset+(start_idx/2)]) <<ENDL();
                        memcpy(&ym1_buffer[((i-1)*BATCH_SIZE_Y)], &in_data[buffer_addr_offset+(start_idx/2)], BATCH_SIZE_Y*2); // Starting from zeroth element, which is boundary condition
                        memcpy(&yp1_buffer[((i-1)*BATCH_SIZE_Y)], &in_data[buffer_addr_offset + 2 + (start_idx/2)], BATCH_SIZE_Y*2); // Starting from second element, which is grid cell plus one
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
