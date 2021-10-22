WARN = -Wmissing-prototypes -Wall #-Winline
OPTI = -O3 -finline-functions -fomit-frame-pointer -DNDEBUG \
-fno-strict-aliasing --param max-inline-insns-single=1800
STD = -std=c99
CC = gcc
MPICC = mpicc
CCFLAGS = $(OPTI) $(WARN) $(STD)

# intel mkl command line: https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-advisor.html
# intel mkl compiler options
MKLOPT = -DMKL_ILP64  -m64  -I"${MKLROOT}/include"
# intel mkl link line
MKLLNK = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lmkl_blacs_openmpi_ilp64 -lpthread -lm -ldl

# fftw link line
FFTWLNK = -lfftw3_mpi -lfftw3 -lm

# gsl link line
GSLLNK  = -lgsl -lgslcblas

.PHONY : help move

help: Makefile
	@sed -n 's/^##//p' $<

## movecomp2: send MPI files to PSMN Comp2
.PHONY : movecomp2
movecomp2:
	scp Makefile comp2:/home/gbritoap/heat3d/
	scp *_mpi.c  comp2:/home/gbritoap/heat3d/
	scp *_mpi.sh comp2:/home/gbritoap/heat3d/
	scp *_mpi.py comp2:/home/gbritoap/heat3d/

## heat: complex heat equation
heat_mpi: heat_mpi.c
	$(MPICC) $(CCFLAGS) $(MKLOPT) -o $@.x $^ $(FFTWLNK) $(MKLLNK)
