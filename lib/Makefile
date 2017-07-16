.PHONY: all clean

.DEFAULT: all

INC = -I$(HOME)/intel/mkl/include
LIBS = -L$(HOME)/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm -lpthread

F90 = gfortran

F90FLAGS = -O3

TARGETS = spconvz.so spconvd.so
TARGETS_F90 = spconvd.f90 spconvz.f90
OBJS = spconvz.o spconvd.o

all: $(TARGETS)

clean:
	rm -f $(TARGETS) $(OBJS) *.mod  $(TARGETS_F90)
spconvz: spconvz.o
	$(F90) -o spconvz spconvz.o $(LIBS) $(INC) $(F90FLAGS)
spconvd: spconvd.o
	$(F90) -o spconvd spconvd.o $(LIBS) $(INC) $(F90FLAGS)
spconvz.so: %: spconvz.f90
	f2py -m spconvz -c spconvz.f90 --fcompiler=gfortran --f90flags="$(F90FLAGS)" $(INC) $(LIBS) -DF2PY_REPORT_ON_ARRAY_COPY=1
spconvd.so: %: spconvd.f90
	f2py -m spconvd -c spconvd.f90 --fcompiler=gfortran --f90flags="$(F90FLAGS)" $(INC) $(LIBS) -DF2PY_REPORT_ON_ARRAY_COPY=1
spconvz.o: spconvz.f90
	$(F90) -c spconvz.f90
spconvd.o: spconvd.f90
	$(F90) -c spconvd.f90
$(TARGETS_F90): spconv.template.f90 frender.py
	python frender.py
