// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include <sys/time.h>
#include <time.h>

using namespace tt;
using namespace tt::tt_metal;

#define ALIGNMENT_PADDING 16

static double getElapsedTime(struct timeval start_time) {
  struct timeval curr_time;
  gettimeofday(&curr_time, NULL);
  long int elapsedtime = (curr_time.tv_sec * 1000000 + curr_time.tv_usec) - (start_time.tv_sec * 1000000 + start_time.tv_usec);
  return elapsedtime / 1000000.0;
}

bfloat16* generate_data(int x, int y) {
    bfloat16 * data= new bfloat16[(x*y)];
    int ctr=1;
    for (int i=0;i<x;i++) {
        for (int j=0;j<y;j++) {
            data[(i*y)+j]=bfloat16((float) 0.0);
        }
    }

     for (int i=0;i<x;i++) {
        // Low value (on LHS)
        data[(i*y)+(ALIGNMENT_PADDING-1)]=bfloat16((float) 1.0);
        // High value (on RHS)
        data[(i*y)+(y-(ALIGNMENT_PADDING))]=bfloat16((float) 20.0);
    }
    return data;
}

void print_host_data(int x, int y, bfloat16 * data) {
    int idx=0;
    for (int i=0;i<x;i++) {
        printf("Row %d:\n", i);
        for (int j=0;j<y;j++) {
            printf("%.1f ", data[idx++].to_float());
        }
        printf("\n");
    }
}

bfloat16* golden_jacobi(bfloat16 * domain_data, int x, int y, int num_its) {
    bfloat16 * uk=new bfloat16[x*y];
    bfloat16 * ukp1=new bfloat16[x*y];

    for (int i=0;i<x;i++) {
        for (int j=0;j<y;j++) ukp1[(i*y)+j]=uk[(i*y)+j]=domain_data[(i*y)+j];
    }

    for (int k=0;k<num_its;k++) {
        for (int i=1;i<x-1;i++) {
            for (int j=ALIGNMENT_PADDING;j<y-ALIGNMENT_PADDING;j++) {
                ukp1[(i*y)+j]=bfloat16((float) 0.25*(uk[((i-1)*y)+j].to_float() + uk[((i+1)*y)+j].to_float() + uk[(i*y)+(j-1)].to_float() + uk[(i*y)+(j+1)].to_float()));
            }
        }
        bfloat16 * tmp=ukp1;
        ukp1=uk;
        uk=tmp;
    }
    delete [] ukp1;
    return uk;
}

void compare_data(bfloat16 * a, bfloat16 * b, int x, int y) {
    int differences=0, checked=0;
    for (int i=0;i<x;i++) {
        for (int j=0;j<y;j++) {
            float val_a=a[(i*y)+j].to_float();
            float val_b=b[(i*y)+j].to_float();
            if(fabs(val_a - val_b) > 0.01){
                differences++;
                printf("Difference at x=%d y=%d, CPU val=%f e150 val=%f\n", i, j, val_a, val_b);
            } else {
                checked++;
            }
        }
    }
    printf("%s: Checked %d grid cells, %d differences\n", differences == 0 ? "Match" : "Error", checked, differences);
}

int main(int argc, char **argv) {

    if (argc !=4) {
        fprintf(stderr, "Must provide three arguments, the problem size in x, problem size in y and number of iterations\n");
        return -1;
    }

    int x_size=atoi(argv[1]);
    int y_size=atoi(argv[2]);
    int num_its=atoi(argv[3]);

    if (x_size%32 != 0 || x_size < 32 || y_size%32 != 0 || y_size < 32) {
        fprintf(stderr, "Domain size must be a multiple of 32, and at-least 32 in size\n");
        return -1;
    }

    printf("Running Jacobi for x=%d, y=%d over %d iterations\n", x_size, y_size, num_its);

    /* Silicon accelerator setup */
    Device *device = CreateDevice(0);

    /* Setup program to execute along with its buffers and kernels to use */
    
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    int total_x_size=x_size+2;
    int total_y_size=y_size+(ALIGNMENT_PADDING*2);

    uint32_t dram_size = total_x_size * total_y_size * 2;
    tt_metal::InterleavedBufferConfig dram_config{
                .device= device,
                .size = dram_size,
                .page_size = dram_size,
                .buffer_type = tt_metal::BufferType::DRAM
    };

    tt_metal::InterleavedBufferConfig dram_config2{
                 .device= device,
                 .size = dram_size,
                 .page_size = dram_size,
                 .buffer_type = tt_metal::BufferType::DRAM
     };

    std::shared_ptr<tt::tt_metal::Buffer> domain_data_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> result_data_dram_buffer = CreateBuffer(dram_config2);

    tt::tt_metal::InterleavedBufferConfig l1_read_config{
         .device= device,
         .size = total_y_size * 2,
         .page_size = total_y_size * 2,
         .buffer_type = tt::tt_metal::BufferType::L1};

    std::shared_ptr<tt::tt_metal::Buffer> l1_read_buffer = CreateBuffer(l1_read_config);

    tt::tt_metal::InterleavedBufferConfig l2_read_config{
         .device= device,
         .size = total_x_size * total_y_size * 2,
         .page_size = total_x_size * total_y_size * 2,
         .buffer_type = tt::tt_metal::BufferType::L1};

    std::shared_ptr<tt::tt_metal::Buffer> l2_read_buffer = CreateBuffer(l2_read_config);

    uint32_t cb_size = 32 * 32 * 2;
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    uint32_t single_tile_size = detail::TileSize(cb_data_format);

    uint32_t src_cb_index = CB::c_in0;
    constexpr uint32_t num_input_tiles = 4;
    CircularBufferConfig cb_input0_config = CircularBufferConfig(num_input_tiles * cb_size, {{src_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src_cb_index, cb_size);
    CBHandle cb_0_src = tt_metal::CreateCircularBuffer(program, core, cb_input0_config);

    src_cb_index = CB::c_in1;
    CircularBufferConfig cb_input1_config = CircularBufferConfig(num_input_tiles * cb_size, {{src_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src_cb_index, cb_size);
    CBHandle cb_1_src = tt_metal::CreateCircularBuffer(program, core, cb_input1_config);

    src_cb_index = CB::c_in2;
    CircularBufferConfig cb_input2_config = CircularBufferConfig(num_input_tiles * cb_size, {{src_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src_cb_index, cb_size);
    CBHandle cb_2_src = tt_metal::CreateCircularBuffer(program, core, cb_input2_config);

    src_cb_index = CB::c_in3;
    CircularBufferConfig cb_input3_config = CircularBufferConfig(num_input_tiles * cb_size, {{src_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src_cb_index, cb_size);
    CBHandle cb_3_src = tt_metal::CreateCircularBuffer(program, core, cb_input3_config);

    constexpr uint32_t output_cb_index = CB::c_out0;
    constexpr uint32_t num_output_tiles = 4;
    CircularBufferConfig cb_output_config = CircularBufferConfig(num_output_tiles * cb_size, {{output_cb_index, tt::DataFormat::Float16_b}}).set_page_size(output_cb_index, cb_size);
    CBHandle cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    src_cb_index = CB::c_intermed0;
    CircularBufferConfig cb_intermediate1_config = CircularBufferConfig(num_input_tiles * cb_size, {{src_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src_cb_index, cb_size);
    CBHandle c_intermed0_src = tt_metal::CreateCircularBuffer(program, core, cb_intermediate1_config);

    src_cb_index = CB::c_intermed1;
    CircularBufferConfig cb_intermediate2_config = CircularBufferConfig(num_input_tiles * cb_size, {{src_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src_cb_index, cb_size);
    CBHandle c_intermed1_src = tt_metal::CreateCircularBuffer(program, core, cb_intermediate2_config);

    //program.allocate_circular_buffers();

    CoreRange all_cores(
        {(std::size_t)0, (std::size_t)0},
        {(std::size_t)0, (std::size_t)0});

    uint32_t semaphore_id = tt_metal::CreateSemaphore(program, all_cores, static_cast<uint32_t>(0));

    bfloat16 * domain_data=generate_data(total_x_size, total_y_size);
    bfloat16 * golden_jacobi_data=golden_jacobi(domain_data, total_x_size, total_y_size, num_its);

    printf("--- LAUNCH ---\n");

    /* Set the parameters that the compute kernel will use */
    vector<uint32_t> compute_kernel_args = {
        static_cast<uint32_t>(x_size),
        static_cast<uint32_t>(y_size),
        static_cast<uint32_t>(num_its),
    };

    /* Use the add_tiles operation in the compute kernel */

    KernelHandle compute_kernel = CreateKernel(
        program,
        "kernels/compute/laplace.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        }
    );

    KernelHandle stream_in_kernel = CreateKernel(
        program,
        "kernels/dataflow/stream_in.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle stream_out_kernel = CreateKernel(
         program,
         "kernels/dataflow/stream_out.cpp",
         core,
         DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    const std::vector<uint32_t> stream_in_runtime_args = {
        domain_data_dram_buffer->address(),
        static_cast<uint32_t>(domain_data_dram_buffer->noc_coordinates().x),
        static_cast<uint32_t>(domain_data_dram_buffer->noc_coordinates().y),
        result_data_dram_buffer->address(),
         static_cast<uint32_t>(result_data_dram_buffer->noc_coordinates().x),
         static_cast<uint32_t>(result_data_dram_buffer->noc_coordinates().y), 
        l1_read_buffer->address(),
        static_cast<uint32_t>(x_size),
        static_cast<uint32_t>(y_size),
        static_cast<uint32_t>(num_its),
        static_cast<uint32_t>(semaphore_id),
    };

    const std::vector<uint32_t> stream_out_runtime_args = {
         domain_data_dram_buffer->address(),
         static_cast<uint32_t>(domain_data_dram_buffer->noc_coordinates().x),
         static_cast<uint32_t>(domain_data_dram_buffer->noc_coordinates().y),
         result_data_dram_buffer->address(),
         static_cast<uint32_t>(result_data_dram_buffer->noc_coordinates().x),
         static_cast<uint32_t>(result_data_dram_buffer->noc_coordinates().y),
         l2_read_buffer->address(),
         static_cast<uint32_t>(x_size),
         static_cast<uint32_t>(y_size),
         static_cast<uint32_t>(num_its),
         static_cast<uint32_t>(semaphore_id),
    };

    SetRuntimeArgs(program, stream_in_kernel, core, stream_in_runtime_args);
    SetRuntimeArgs(program, stream_out_kernel, core, stream_out_runtime_args);

    bfloat16 * device_results=generate_data(total_x_size, total_y_size);

    struct timeval start_time;
    gettimeofday(&start_time, NULL);
    EnqueueWriteBuffer(cq, domain_data_dram_buffer, domain_data, false);
    EnqueueWriteBuffer(cq, result_data_dram_buffer, device_results, false);
    double xfer_on_time=getElapsedTime(start_time);

    gettimeofday(&start_time, NULL); 
    EnqueueProgram(cq, program, true);
    double exec_time=getElapsedTime(start_time);

    printf("--- COMPLETED RUN ---\n");

    // Minus one here as its counts from zero on the device
    gettimeofday(&start_time, NULL);
    if ((num_its-1) % 2 == 0) {
        EnqueueReadBuffer(cq, result_data_dram_buffer, device_results, true);
    } else {
        EnqueueReadBuffer(cq, domain_data_dram_buffer, device_results, true);
    }
    double xfer_off_time=getElapsedTime(start_time);
    printf("--- COPIED DATA BACK ---\n");
    //compare_data(device_results, golden_jacobi_data, total_x_size, total_y_size);
    //print_host_data(total_x_size, total_y_size, device_results);
    //print_host_data(total_x_size, total_y_size, golden_jacobi_data);
    CloseDevice(device);

    double total_time=xfer_on_time+exec_time+xfer_off_time;
    printf("Run completed, total time %.4f sec. %.4f sec transfer on, %.4f sec execution, %.4f sec transfer off\n", total_time, xfer_on_time, exec_time, xfer_off_time); 

    unsigned long total_points=x_size * y_size * num_its;
    printf("Execution GPts: %.4f\n", (total_points / exec_time)/(1024*1024*1024));
    printf("Total GPts: %.4f\n", (total_points / total_time)/(1024*1024*1024));
}
