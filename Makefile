main: main.cpp mixedDpll.o bruteForce.o 
	g++ -std=c++11 -o main -L/usr/local/depot/cuda-10.2/lib64/ -lcudart main.cpp mixed_dpll/mixedDpll.o brute_force/bruteForce.o 

clean:
	rm main mixed_dpll/*.o brute_force/*.o

bruteForce.o: brute_force/bruteForce.cu
	nvcc -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -c brute_force/bruteForce.cu
	mv bruteForce.o brute_force/

mixedDpll.o: mixed_dpll/mixedDpll.cu
	nvcc -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -c mixed_dpll/mixedDpll.cu
	mv mixedDpll.o mixed_dpll/

bruteForce: brute_force/bruteForce.cu
	nvcc brute_force/bruteForce.cu -o bruteForceParallel


