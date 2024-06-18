CXX=clang-17
CFLAGS=-O3 -Wno-int-to-pointer-cast -mavx2 -DFMT_HEADER_ONLY -fvisibility-inlines-hidden -fno-lto -DTT_METAL_VERSIM_DISABLED -Werror -Wdelete-non-virtual-dtor -Wreturn-type -Wswitch -Wuninitialized -Wno-unused-parameter -Wsometimes-uninitialized -Wno-c++11-narrowing -Wno-c++20-extensions -Wno-c++23-extensions -Wno-error=local-type-template-args -Wno-delete-non-abstract-non-virtual-dtor -Wno-c99-designator -Wno-shift-op-parentheses -Wno-non-c-typedef-for-linkage -std=c++17
INCLUDE=-I${TT_METAL_HOME}/tt_metal/third_party/umd -I${TT_METAL_HOME}/tt_metal/third_party/fmt -I${TT_METAL_HOME}/tt_metal/hw/inc/grayskull/ -I${TT_METAL_HOME}/tt_metal/third_party/umd/src/firmware/riscv/grayskull -I${TT_METAL_HOME}/tt_metal/hw/inc -I${TT_METAL_HOME}/tt_metal -I${TT_METAL_HOME}

LINKER=/usr/bin/c++
LFLAGS=-rdynamic -L${TT_METAL_LIB} -ltt_metal -ldl -lstdc++fs -pthread -lyaml-cpp -lm

all:
	${CXX} ${CFLAGS} ${INCLUDE} -c jacobi_in_riscv.cpp
	${LINKER} jacobi_in_riscv.o -o jacobi ${LFLAGS}
