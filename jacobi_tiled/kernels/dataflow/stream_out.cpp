#include <cstdint>
#include <string.h>

// Byte alignment
#define ALIGNMENT 32
// Print debug statements
#define PRINT_DEBUG 0

#define BATCH_SIZE_X 32
#define BATCH_SIZE_Y 32

void kernel_main() {
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

    std::uint32_t total_x_size = x_size;
    std::uint32_t total_y_size = y_size+((ALIGNMENT/2)*2);  // Number of elements is alignment div 2, and we have LHS and RHS

    std::uint32_t num_batches_x=x_size/BATCH_SIZE_X;
    std::uint32_t num_batches_y=y_size/BATCH_SIZE_Y;

    constexpr uint32_t cb_id_out0 = tt::CB::c_out0;    

    volatile tt_l1_ptr uint32_t* semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);

    uint64_t semaphore_noc_addr=get_noc_addr(0, 0, semaphore_addr);

    for (uint32_t it=0;it<num_its;it++) {
        noc_semaphore_set(semaphore_addr_ptr, it);
#if PRINT_DEBUG==1        
        DPRINT << "Start data writer for iteration " << U32(it) << ENDL();
#endif        

        for (uint32_t bx=0;bx<num_batches_x;bx++) {
            std::uint32_t batch_x_start=bx*BATCH_SIZE_X*total_y_size;
            for (uint32_t by=0;by<num_batches_y;by++) {
                std::uint32_t batch_y_start=by*BATCH_SIZE_Y;
                cb_wait_front(cb_id_out0, 1);
                std::uint32_t data_buffer_addr = get_read_ptr(cb_id_out0);
                for (uint32_t i=0;i<BATCH_SIZE_X;i++) {
                    // Add one to start writing from second line
                    uint32_t dram_addr_offset=((batch_x_start+((i+1)*total_y_size)+batch_y_start)*2) + ALIGNMENT;
                    uint32_t buffer_addr_offset=(i*BATCH_SIZE_Y)*2;

                    uint64_t domain_data_buffer_noc_addr;
                    if (it % 2 == 0) {
                        domain_data_buffer_noc_addr = get_noc_addr(area_two_data_buffer_noc_x, area_two_data_buffer_noc_y, area_two_data_buffer_addr+dram_addr_offset);
                    } else {
                        domain_data_buffer_noc_addr = get_noc_addr(area_one_data_buffer_noc_x, area_one_data_buffer_noc_y, area_one_data_buffer_addr+dram_addr_offset);
                    }
                    noc_async_write(data_buffer_addr+buffer_addr_offset, domain_data_buffer_noc_addr, BATCH_SIZE_Y*2);
                    noc_async_write_barrier();
                }

                cb_pop_front(cb_id_out0, 1);
            }
        }
#if PRINT_DEBUG==1        
        DPRINT << "Finish data writer for iteration " << U32(it) << ENDL();
#endif        
    }
#if PRINT_DEBUG==1    
    DPRINT << "Leaving stream out" << ENDL();
#endif    
}
