CC = gcc
OPTS = -O3

CFITSIO_DIR = /usr/local

CFITSIO_INCDIR=$(CFITSIO_DIR)/include/cfitsio
CFITSIO_LIBDIR=$(CFITSIO_DIR)/lib 

EXT_LIBS = -lm -L$(CFITSIO_DIR)/lib -lcfitsio

CFLAGS = $(OPTS)

EXEC = generateRays

HEALPIX_LIB = chealpix/libchealpix.a

OBJS = $(EXEC).o

LOCAL_LIBS = $(HEALPIX_LIB)

all: $(EXEC)

$(EXEC): $(OBJS) $(LOCAL_LIBS)
	$(CC) $(CFLAGS) -o $(EXEC) $(OBJS) $(LOCAL_LIBS) $(EXT_LIBS)

$(HEALPIX_LIB):
	make CC=$(CC) OPT=$(OPTS) CFITSIO_INCDIR=$(CFITSIO_INCDIR) CFITSIO_LIBDIR=$(CFITSIO_LIBDIR)-C chealpix

clean:
	-rm *.o
	-rm Make.log
	make clean -C chealpix

distclean: clean
	-rm $(EXEC)
	make distclean -C chealpix
	
