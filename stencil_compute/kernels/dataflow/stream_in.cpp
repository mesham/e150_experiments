#include <cstdint>
#include <string.h>

void kernel_main() {
    std::uint32_t domain_data_buffer_addr  = get_arg_val<uint32_t>(0);
    std::uint32_t domain_data_buffer_noc_x = get_arg_val<uint32_t>(1);
    std::uint32_t domain_data_buffer_noc_y = get_arg_val<uint32_t>(2);

    std::uint32_t read_in_buffer_addr = get_arg_val<uint32_t>(3);

    std::uint32_t x_size  = get_arg_val<uint32_t>(4);
    std::uint32_t y_size  = get_arg_val<uint32_t>(5);

    std::uint32_t total_x_size = x_size + 2;
    std::uint32_t total_y_size = y_size + 2;

    const uint32_t type_size = 2;

    std::uint32_t data_size=total_x_size*total_y_size*type_size;

    char* in_data=(char*) read_in_buffer_addr;

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_id_in1 = tt::CB::c_in1;
    constexpr uint32_t cb_id_in2 = tt::CB::c_in2;
    constexpr uint32_t cb_id_in3= tt::CB::c_in3;

    DPRINT << "Begin data load" << ENDL();
    

    const InterleavedAddrGen<true> s0 = {
          .bank_base_address = domain_data_buffer_addr,
          .page_size = total_y_size*type_size,
    };

    cb_reserve_back(cb_id_in0, 1);
    cb_reserve_back(cb_id_in1, 1);
    cb_reserve_back(cb_id_in2, 1);
    cb_reserve_back(cb_id_in3, 1);

    char * yp1_buffer = (char*) get_write_ptr(cb_id_in0);
    char * ym1_buffer = (char*) get_write_ptr(cb_id_in1);
    char * xp1_buffer = (char*) get_write_ptr(cb_id_in2);
    char * xm1_buffer = (char*) get_write_ptr(cb_id_in3);

    for (uint32_t i=0;i<total_x_size;i++) {
        noc_async_read_page(i, s0, read_in_buffer_addr);
        noc_async_read_barrier();
        //DPRINT << "Start it " << U32(i) << ENDL();
        //for (uint32_t j=0;j<total_y_size;j++) {
        //    ((uint16_t*)in_data)[j]=(uint16_t) 10.0;
        //    DPRINT << "Val " << U32(i) << "," << U32(j) << ": "<< BF16(((uint16_t*)in_data)[j]) << ENDL();
        //}
        if (i < x_size) {
            memcpy(xm1_buffer, &in_data[type_size], y_size*type_size);
        }
        if (i >= 2) {
            memcpy(&xp1_buffer[((i-2)*y_size)*type_size], &in_data[type_size], y_size*type_size);
        }
        if (i >=1 && i < x_size+1) {
            // For now, lets have this the same for testing
            memcpy(&yp1_buffer[((i-1)*y_size)*type_size], in_data, y_size*type_size);
            memcpy(&ym1_buffer[((i-1)*y_size)*type_size], in_data, y_size*type_size);
            //memcpy(&yp1_buffer[((i-1)*y_size)*type_size], &in_data[2*type_size], y_size*type_size);
            //memcpy(&ym1_buffer[((i-1)*y_size)*type_size], &in_data[0], y_size*type_size);
        }
        //DPRINT << "End it " << U32(i) << ENDL();
    }

    cb_push_back(cb_id_in0, 1);
    cb_push_back(cb_id_in1, 1);
    cb_push_back(cb_id_in2, 1);
    cb_push_back(cb_id_in3, 1);
   
    DPRINT << "Done data load" << ENDL();
}
