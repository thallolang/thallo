EXECUTABLE = deconvolution
OBJS = build/mLibSource.o build/main.o

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
  LFLAGS += -L../external/FreeImage/lib/osx -Wl,-rpath,../external/FreeImage/lib/osx
endif

ifeq ($(UNAME), Linux)
  LFLAGS += -L../external/FreeImage/lib/linux -Wl,-rpath,../external/FreeImage/lib/linux
endif

LFLAGS += -lfreeimage

USES_CERES=1

include ../shared/make_template.inc

