main: main.cpp bruteForce.o
	g++ -std=c++11 -o main -L/usr/local/depot/cuda-10.2/lib64/ -lcudart main.cpp brute_force/bruteForce.o

clean:
	rm main *.o

bruteForce.o: brute_force/bruteForce.cu
	nvcc -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -c brute_force/bruteForce.cu
	mv bruteForce.o brute_force/

bruteForce: brute_force/bruteForce.cu
	nvcc brute_force/bruteForce.cu -o bruteForceParallel


