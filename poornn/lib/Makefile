.PHONY: all clean

.DEFAULT: all

INC = -I$(HOME)/intel/mkl/include
LIBS = -L$(HOME)/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm -lpthread

F90 = gfortran

F90FLAGS = -O3

SOURCES = templates/spconv.template.f90 templates/linear.template.f90 templates/pooling.template.f90 templates/relu.template.f90
TARGETS_F90 = spconv.f90 linear.f90 pooling.f90 relu.f90
OBJS = $(TARGETS_F90:.f90=.o)
TARGETS = $(TARGETS_F90:.f90=.so)

all: $(TARGETS)

clean:
	rm -f $(TARGETS) $(OBJS) *.mod  $(TARGETS_F90)
spconv: spconv.o
	$(F90) -o spconv spconv.o $(LIBS) $(INC) $(F90FLAGS)
spconv.so: %: spconv.f90
	f2py -m spconv -c spconv.f90 --fcompiler=gfortran --f90flags="$(F90FLAGS)" $(INC) $(LIBS) -DF2PY_REPORT_ON_ARRAY_COPY=1
linear.so: %: linear.f90
	f2py -m linear -c linear.f90 --fcompiler=gfortran --f90flags="$(F90FLAGS)" $(INC) $(LIBS) -DF2PY_REPORT_ON_ARRAY_COPY=1
pooling.so: %: pooling.f90
	f2py -m pooling -c pooling.f90 --fcompiler=gfortran -DF2PY_REPORT_ON_ARRAY_COPY=1
relu.so: %: relu.f90
	f2py -m relu -c relu.f90 --fcompiler=gfortran -DF2PY_REPORT_ON_ARRAY_COPY=1
spconv.o: spconv.f90
	$(F90) -c spconv.f90
linear.o: linear.f90
	$(F90) -c linear.f90
pooling.o: pooling.f90
	$(F90) -c pooling.f90
relu.o: relu.f90
	$(F90) -c relu.f90
$(TARGETS_F90): $(SOURCES) frender.py
	python frender.py
