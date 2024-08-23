CXX=clang++-17
CFLAGS=-I${TT_METAL_HOME}/tt_metal/third_party/umd -I${TT_METAL_HOME} -I${TT_METAL_HOME}/tt_metal -I${TT_METAL_HOME}/tt_metal/common -I${TT_METAL_HOME}/tt_metal/programming_examples -I${TT_METAL_HOME}/tt_metal/third_party/fmt -I${TT_METAL_HOME}/tt_metal/hw/inc -I${TT_METAL_HOME}/tt_metal/hw/inc/grayskull -I${TT_METAL_HOME}/tt_metal/third_party/umd/device/grayskull -I${TT_METAL_HOME}/tt_metal/third_party/umd/src/firmware/riscv/grayskull -isystem ${TT_METAL_HOME}/.cpmcache/reflect/e75434c4c5f669e4a74e4d84e0a30d7249c1e66f -isystem ${TT_METAL_HOME}/.cpmcache/flatbuffers/2c4062bffa52fa4157b1b4deeae73395df475fda/include -isystem ${TT_METAL_HOME}/.cpmcache/nanomsg/28cc32d5bdb6a858fe53b3ccf7e923957e53eada/include -O3 -Wno-int-to-pointer-cast -stdlib=libc++ -mavx2 -fPIC -DFMT_HEADER_ONLY -fvisibility-inlines-hidden -fno-lto -DARCH_GRAYSKULL -DDISABLE_ISSUE_3487_FIX -Werror -Wdelete-non-virtual-dtor -Wreturn-type -Wswitch -Wuninitialized -Wno-unused-parameter -Wsometimes-uninitialized -Wno-c++11-narrowing -Wno-c++23-extensions -Wno-error=local-type-template-args -Wno-delete-non-abstract-non-virtual-dtor -Wno-c99-designator -Wno-shift-op-parentheses -Wno-non-c-typedef-for-linkage -Wno-deprecated-this-capture -Wno-deprecated-volatile -Wno-deprecated-builtins -Wno-deprecated-declarations -std=c++20
INCLUDE=-I${TT_METAL_HOME}/tt_metal/third_party/umd -I${TT_METAL_HOME}/tt_metal/third_party/fmt -I${TT_METAL_HOME}/tt_metal/hw/inc/grayskull/ -I${TT_METAL_HOME}/tt_metal/third_party/umd/src/firmware/riscv/grayskull -I${TT_METAL_HOME}/tt_metal/hw/inc -I${TT_METAL_HOME}/tt_metal -I${TT_METAL_HOME} -I${TT_METAL_HOME}/include
INCLUDE=-I${TT_METAL_HOME}/tt_metal/third_party/umd -I${TT_METAL_HOME}/tt_metal/third_party/fmt -I${TT_METAL_HOME}/tt_metal/hw/inc/grayskull/ -I${TT_METAL_HOME}/tt_metal/third_party/umd/src/firmware/riscv/grayskull -I${TT_METAL_HOME}/tt_metal/hw/inc -I${TT_METAL_HOME}/tt_metal -I${TT_METAL_HOME} -I${TT_METAL_HOME}/include

LINKER=clang++-17
LFLAGS=-rdynamic -L${TT_METAL_HOME}/build/lib -ltt_metal -ldl -lstdc++fs -pthread  -lm -lc++

all:
	${CXX} ${CFLAGS} -c jacobi.cpp
	${LINKER} jacobi.o -o jacobi ${LFLAGS}
