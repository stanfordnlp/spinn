testrun:
	nvcc -g -O0 -lcublas -lcurand \
	  -std=c++11 -o testrun \
	  kernels.cu testrun.cc thin-stack.cc util.cc

clean:
	rm -f testrun
	rm -f *.o *.cu_o
