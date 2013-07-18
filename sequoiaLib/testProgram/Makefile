CXX = mpicxx
CC  = mpicc
LD  = mpicxx 
F90  = ifort

.SUFFIXES: .o .cpp .ptx .cu

CUDA_TK  = /usr/local/cuda

# OFLAGS = -O0 -g -Wall 
OFLAGS = -O3 -g -Wall -fopenmp 
# OFLAGS = -O3  -Wall -fopenmp 
CXXFLAGS =  -fPIC $(OFLAGS) -I$(CUDA_TK)/include 

# Use with Mac OS X
# NVCCFLAGS = -arch sm_12 -Xcompiler="-Duint=unsigned\ int"

LDFLAGS = -lcuda  -lsequoia -L../lib
#LDFLAGS = -lsequoia -L../lib



#For tipsy support uncomment the following line
#CXXFLAGS += -DTIPSYOUTPUT


#INCLUDEPATH = ./include
#CXXFLAGS  += -I$(INCLUDEPATH)


SRCPATH =
SRC = main.cpp
OBJ = $(SRC:%.cpp=%.o)

PROG = main


all:	$(OBJ)  $(PROG)
	ln -s ../lib/*.ptx . -f




$(PROG): $(OBJ)
	$(LD) $^ -o $@ $(LDFLAGS)

%.o: $(SRCPATH)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	/bin/rm -rf *.o *.ptx main *.a



