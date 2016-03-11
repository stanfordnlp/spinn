test:
	nvcc -g -O0 -lcublas -std=c++11 -o test kernels.cu test.cc thin-stack.cc util.cc

clean:
	rm test
	rm -f *.o *.cu_o
