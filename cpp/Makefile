GTEST_DIR = lib/googletest/googletest
GMOCK_DIR = lib/googletest/googlemock
CPPFLAGS += -isystem $(GTEST_DIR)/include -isystem $(GMOCK_DIR)/include
CXXFLAGS += -g -Wall -Wextra -pthread

######
# gtest directives
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)
gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest-all.cc

gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest_main.cc

gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

GMOCK_SRCS_ = $(GMOCK_DIR)/src/*.cc $(GMOCK_HEADERS)
gmock-all.o : $(GMOCK_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) -I$(GMOCK_DIR) $(CXXFLAGS) \
	  -c $(GMOCK_DIR)/src/gmock-all.cc

gmock_main.o : $(GMOCK_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) -I$(GMOCK_DIR) $(CXXFLAGS) \
	  -c $(GMOCK_DIR)/src/gmock_main.cc

gmock.a : gmock-all.o gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gmock_main.a : gmock-all.o gtest-all.o gmock_main.o
	$(AR) $(ARFLAGS) $@ $^
######

SOURCES = thin-stack.cc rnn.cc \
		  kernels.cu util.cc

test: gtest_main.a gmock_main.a test.cc $(SOURCES)
	rm -f bin/unittest
	nvcc -g -O0 -lcublas -lcurand \
		-isystem $(GTEST_DIR)/include -isystem $(GMOCK_DIR)/include \
		-std=c++11 -o bin/unittest \
		$^

stacktest: bin/stacktest.cc $(SOURCES)
	nvcc -g -O0 -lcublas -lcurand \
	  -std=c++11 -I. -o bin/stacktest \
	  $^

rnntest: bin/rnntest.cc $(SOURCES)
	nvcc -g -O0 -lcublas -lcurand \
	  -std=c++11 -I. -o bin/rnntest \
	  $^

spinn: bin/spinn.cc $(SOURCES)
	nvcc -g -O0 -lcublas -lcurand \
		-std=c++11 -I. -o bin/spinn \
		$^

clean:
	rm -f bin/unittest bin/stacktest bin/rnntest
	rm -f *.a *.o *.cu_o
