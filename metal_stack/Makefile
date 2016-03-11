test:
	nvcc -g -O0 -lcublas -lcurand \
	  -std=c++11 -o test \
	  kernels.cu test.cc thin-stack.cc util.cc

clean:
	rm -f test
	rm -f *.o *.cu_o
