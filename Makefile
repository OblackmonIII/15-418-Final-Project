main: main.cpp bruteForce.o
	g++ -std=c++11 -o main -L/usr/local/depot/cuda-10.2/lib64/ -lcudart main.cpp bruteForce.o

clean:
	rm main *.o

bruteForce.o: bruteForce.cu
	nvcc -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -c bruteForce.cu

bruteForce: bruteForce.cu
	nvcc bruteForce.cu -o bruteForceParallel


