CXX       := g++
CXX_FLAGS := -std=c++11 -ggdb -g -fopenmp -Wall -w -mavx512f -msse4.2 -march=native
RELEASE_CXX_FLAGS := -std=c++11 -o3 -fopenmp -mavx512f -msse4.2 -march=native

BIN     := 
SRC     := ./*.cpp
INCLUDE := 

LIBRARIES  := 
# -L/opt/intel/oneapi/mkl/2022.2.0/lib/intel64 -lmkl_rt \

EXECUTABLE  := main


all: clean
	$(CXX) $(CXX_FLAGS) $(INCLUDE) $(SRC) -o $(EXECUTABLE) $(LIBRARIES)

release: clean
	$(CXX) $(RELEASE_CXX_FLAGS) $(INCLUDE) $(SRC) -o $(EXECUTABLE) $(LIBRARIES)

run:
	./$(EXECUTABLE)

# test:
# 	./$(EXECUTABLE) -m matrix_case/benchmark/oscil_trans_01.mtx -p 2 -s 6


clean:
	rm -rf *.o $(EXECUTABLE)
