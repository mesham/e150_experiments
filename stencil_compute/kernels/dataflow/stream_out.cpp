#include <cstdint>
#include <string.h>

void kernel_main() {
    std::uint32_t result_data_buffer_addr  = get_arg_val<uint32_t>(0);
    std::uint32_t result_data_buffer_noc_x = get_arg_val<uint32_t>(1);
    std::uint32_t result_data_buffer_noc_y = get_arg_val<uint32_t>(2);

    std::uint32_t x_size  = get_arg_val<uint32_t>(3);
    std::uint32_t y_size  = get_arg_val<uint32_t>(4);

    const uint32_t type_size = 2;

    constexpr uint32_t cb_id_out0 = tt::CB::c_out0;

    uint64_t results_data_buffer_noc_addr = get_noc_addr(result_data_buffer_noc_x, result_data_buffer_noc_y, result_data_buffer_addr);

    //const DataFormat data_format = get_dataformat(cb_id_out0);

    //const InterleavedAddrGenFast<true> s0 = {
    //    .bank_base_address = result_data_buffer_addr,
    //    .page_size = (y_size)*type_size,
    //    .data_format = data_format,
    //};

    DPRINT << "Begin data writer " << U32(cb_id_out0) << ENDL();

    cb_wait_front(cb_id_out0, 1);
    DPRINT << "Got output, proceeding " << ENDL();
    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

    uint16_t * in_data=(uint16_t*) l1_read_addr;

    //for (uint32_t i=0;i<x_size;i++) {
    //    noc_async_write((uint32_t) &in_data[i*y_size], results_data_buffer_noc_addr, y_size*2);
    //    noc_async_write_barrier();
    //    results_data_buffer_noc_addr+=(y_size+2)*2;
    //}

    noc_async_write(l1_read_addr, results_data_buffer_noc_addr, x_size*y_size*2);
    noc_async_write_barrier();

    cb_pop_front(cb_id_out0, 1);
    DPRINT << "Finish data writer " << ENDL();
}
