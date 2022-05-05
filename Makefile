main: main.cpp bruteForce.o mixed_dpll.o
	g++ -std=c++11 -o main -L/usr/local/depot/cuda-10.2/lib64/ -lcudart main.cpp mixed_dpll/mixed_dpll.o brute_force/bruteForce.o 

clean:
	rm main *.o

bruteForce.o: brute_force/bruteForce.cu
	nvcc -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -c brute_force/bruteForce.cu
	mv bruteForce.o brute_force/

mixed_dpll.o: mixed_dpll/mixed_dpll.cu
	nvcc -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -c mixed_dpll/mixed_dpll.cu
	mv mixed_dpll.o mixed_dpll/

bruteForce: brute_force/bruteForce.cu
	nvcc brute_force/bruteForce.cu -o bruteForceParallel


