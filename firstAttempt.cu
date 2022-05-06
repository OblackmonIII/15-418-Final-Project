#include <stdio.h>
#include "firstAttempt.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"


__device__ int dpllParallel(std::vector<set<int> > clauses, int nvars){
    // Return 1 if satisfiable
    // Return -1 if unsatisfiable
    int result = 0;
    while(unit_clauses_present(clauses) && clauses.size() > 1){
        int unit_clause_variable = find_unit_clause(clauses);
        clauses = unit_propagate(clauses, clauses.size(), unit_clause_variable);
    }

    while(pure_literals_present(clauses, nvars)){
        int pure_literal = find_pure_literal(clauses, nvars);
        clauses = pure_literal_assign(clauses, pure_literal);
    }

    // Check if there are no more clauses
    if(clauses.size() == 0){
        return 1;
    }

    // Check for empty clauses
    if(check_for_empty_clauses(clauses)){
        return 0;
    }

    set<int> literals = get_literals(clauses);

    int literal_choice = *(literals.begin());
    set<int> new_set;
    new_set.insert(literal_choice);
    clauses.push_back(new_set);
    result = dpllParallel(clauses, clauses.size());

    if(result == 0){
        clauses[clauses.size() - 1].erase(literal_choice);
        clauses[clauses.size() - 1].insert(-1 * literal_choice);
        result = dpllParallel(clauses, clauses.size());
    }

    return result;
    
}

__global__ void brute_force_kernel(int *clauses_arr_device, int *clauses_length_arr_device, int nclauses, int *var_assignment_output_device, int num_of_assignments){

    int satResult = dpllParallel(clauses, nvars);

    var_assignment_output_device[threadId] = satResult;
}

double cudaDpll(int *inarray, int *end, int *resultarray) {
    int *device_data;

    cudaMalloc((void **)&device_data, sizeof(int) * rounded_length);

    cudaMemcpy(device_data, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_data, end - inarray);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(resultarray, device_data, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);
    return overallDuration;
}

int firstAttempt::initialDpllParallel(std::vector<std::set<int> > clauses, int nvars){
	int result = 0;

    dpll_first_kernel<<<1, 1>>>(clauses_arr_device, clauses_length_arr_device, clauses.size(), var_assignment_output_device, num_of_assignments);

    // Synchronize threads
    cudaDeviceSynchronize();

    // Move values back to host code
    cudaMemcpy(var_assignment_output_host, var_assignment_output_device, sizeof(int) * num_of_assignments, cudaMemcpyDeviceToHost);

    
    // Free host memory
    free(clauses_arr_host);
    free(clauses_length_arr_host);
    free(var_assignment_output_host);

    // Free device memory
    cudaFree(clauses_arr_device);
    cudaFree(clauses_length_arr_device);
    cudaFree(var_assignment_output_device);

    return result;
}