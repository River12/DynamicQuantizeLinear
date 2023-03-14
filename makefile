CXX       := g++
CXX_FLAGS := -std=c++11 -ggdb -g -fopenmp -Wall -w -mavx512f -march=native
RELEASE_CXX_FLAGS := -std=c++11 -o3 -fopenmp -mavx512f -march=native

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

clean:
	rm -rf *.o $(EXECUTABLE)
