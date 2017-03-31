INCLUDES=../utils
CXXFLAGS=-std=c++11 -Wno-deprecated-gpu-targets

all: main run

main: main.cu
	nvcc -I$(INCLUDES) $(CXXFLAGS) main.cu lab1.cu -o main
run: main
	./main
clean:
	rm main
