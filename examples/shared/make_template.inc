CERES_INC=/usr/local/include/ceres/include
CERES_LIB=/usr/local/include/ceres/lib
CERES_ENABLED=1

CXX = clang++
LINK = clang++
NVCC = /usr/local/cuda/bin/nvcc
SRC ?= src
CUDAHOME += /usr/local/cuda
TERRAHOME ?= ../../../terra
TERRA ?= ../../../terra/bin/terra

ifdef USES_CERES
  ifeq ($(CERES_ENABLED),1)
    LFLAGS += -lglog -lceres -lumfpack -lamd -lcholmod -lsuitesparseconfig -lcxsparse -lblas -llapack -lomp
    LFLAGS += -L$(CERES_LIB)
    FLAGS += -I$(CERES_INC) 
  endif
endif

FLAGS += -O3 -g -I$(SRC) -I$(SRC)/cutil/inc -I../../API/release/include -I$(TERRAHOME)/include -I../external/mLib/include -I../external -I../external/OpenMesh/include -I../external/FreeImage/include -I../shared -I/usr/include/eigen3 -I../shared/tclap/include
NVCCFLAGS += -arch sm_30 -std=c++11 $(FLAGS)
LFLAGS += -std=c++11 -L../../API/release/lib -L$(TERRAHOME)/lib -lterra -ldl
UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
  LFLAGS += -pagezero_size 10000 -image_base 100000000 -lz -lcurses
  FLAGS += -DTHALLO_CPU=1
endif

ifeq ($(UNAME), Linux)
  S4 = 64
  LFLAGS += -Wl,-rpath,$(CUDAHOME)/lib$(S4) -pthread
  LFLAGS += -L$(CUDAHOME)/lib$(S4)
  LFLAGS += -lcudart
  FLAGS += -I$(CUDAHOME)/include
endif

.SUFFIXES:
.PHONY:	all purge clean

all: $(EXECUTABLE)

THALLOLIB = ../../API/release/lib/libThalloDev.a

build/%.o:	$(SRC)/%.cpp
	@mkdir -p $(@D)
	$(CXX) -std=c++14 $(FLAGS) -MP -MD $< -c -o $@
build/%.o:	$(SRC)/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

$(THALLOLIB):
	$(MAKE) -C ../../API/

$(EXECUTABLE):	$(OBJS) $(THALLOLIB)
	$(CXX) $^ -o $@ $(LFLAGS)

clean:
	rm -rf build/*.o build/*.d
	rm -rf $(EXECUTABLE)

purge: clean
	rm -rf build/*

# dependency rules
include $(wildcard build/*.d)
