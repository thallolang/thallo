.PHONY:	all clean purge release

TERRAHOME ?= ../../terra/
TERRA ?= ../../terra/bin/terra

all: release/lib/libThallo.a release/lib/libThalloDev.a

release/lib/libThallo.a:	$(wildcard src/*.t)
	@mkdir -p release/lib
	$(TERRA) src/createwrapper.t Thallo src thallo release/include/Thallo.h build/wrapper.o true
	rm -rf $@
	ar cq $@ build/wrapper.o

release/lib/libThalloDev.a:	$(wildcard src/*.t)
	@mkdir -p release/lib
	$(TERRA) src/createwrapper.t Thallo src thallo release/include/Thallo.h build/wrapperdev.o false
	rm -rf $@
	ar cq $@ build/wrapperdev.o

clean:
	rm -rf build/wrapper.o
	rm -rf release/lib/libThallo.a
	rm -rf release/lib/libThalloDev.a
	rm -rf release/lib/libterra.a