#include "bruteForce.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <stdio.h>

__device__ int get_nth_bit(int value, int n){

    return (value >> n) & (1);
}

__device__ bool test_assignment(int var_assignment_val, int *clauses, int *clause_length_arr, int nclauses){

    // Iterate over clauses
    // Assign variables found in clauses based on array
    // If the clause returns false, return 0
    // Otherwise continue to the next clause
    // If all clauses return true, return true

    bool result = false;
    bool var_assignment;
    int starting_index = 0;
    int nth_bit;
    for(int i = 0; i < nclauses; i++){
        
        result = false;

        for(int j = starting_index; j < starting_index + clause_length_arr[i]; j++){
            // Take absolute value of the variable
            // if the value is false, negate the assignment
            // otw, leave it the same             
            int index = std::abs(clauses[j]) - 1;
            nth_bit = get_nth_bit(var_assignment_val, index);
            var_assignment =  nth_bit == 0 ? false : true;


            if(clauses[j] > 0){

                result = result || var_assignment;
            }else{

                result = result || !var_assignment;
            }

        } 

        starting_index += clause_length_arr[i];
        
        if(!result){


            return result;
        }

    }

    return result;
}

__global__ void brute_force_kernel(int *clauses_arr_device, int *clauses_length_arr_device, int nclauses, int *var_assignment_output_device, int num_of_assignments){

    // Implementation ideas
    /* 
        A specific thread with threadId x will have its binary representation correspond to a variable assignment
    - Use that to construct a variable assignment array
    - Have the clause structure exist in the shared memory
    - Have a kernel function that evaluates the assignment for a given set of clauses
    - Store assignment evaluation value in the return array
    - Scan over this return array and if 1 is an output, then the expression is satisfiable.
    - Unsatisfiable otherwise
    */

    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int assignment_result = test_assignment(threadId, clauses_arr_device, clauses_length_arr_device, nclauses) ? 1 : 0;

    var_assignment_output_device[threadId] = assignment_result;

}

struct is_one
{
  __host__ __device__
  bool operator()(int x)
  {
    return x == 1;
  }
};

__host__ int *clauses_to_array_b(std::vector<std::set<int> > clauses, int sum_length_of_clauses){

    int *clauses_arr = (int *) malloc(sum_length_of_clauses * sizeof(int));
    int clause_offset = 0;
    int j = 0;

    for(int i = 0; i < clauses.size(); i++){
        
        j = 0;
        //clauses_arr[i] = (int *) malloc(clauses[i].size() * sizeof(int));
        std::set<int>::iterator pure_literal_itr;
        for(pure_literal_itr = clauses[i].begin(); pure_literal_itr != clauses[i].end(); pure_literal_itr++){

            clauses_arr[j + clause_offset] = *pure_literal_itr;
            j++;
        }

        clause_offset += clauses[i].size();
    }


    return clauses_arr;
}

int *get_clauses_length_arr_b_b(std::vector<std::set<int> > clauses){

    int *clauses_length_arr = (int *) malloc(clauses.size() * sizeof(int));

    for(int i = 0; i < clauses.size(); i++){

        clauses_length_arr[i] = clauses[i].size();
    }

    return clauses_length_arr;
}

int get_sum_length_of_clauses_b(std::vector<std::set<int> > clauses){

    int sum = 0;

    for(int i = 0; i < clauses.size(); i++){

        sum += clauses[i].size();
    }


    return sum;
}

int BruteForce::brute_force_parallel(std::vector<std::set<int> > clauses, int nvars){

    // Number of assignments
    int num_of_assignments = (double) pow(2.0, (double) nvars);

    // Allocate memory for the clauses on the host, fill with clause arrays, and move to the device
    int sum_length_of_clauses = get_sum_length_of_clauses_b(clauses);
    int *clauses_arr_host = clauses_to_array_b(clauses, sum_length_of_clauses);

    int *clauses_arr_device;
    // Allocate memory for the clauses array
    cudaMalloc(&clauses_arr_device, sum_length_of_clauses * sizeof(int));

    // Copy values to device
    cudaMemcpy(clauses_arr_device, clauses_arr_host, sum_length_of_clauses * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate memory that maps clause index to the number of variables in the clause
    int *clauses_length_arr_host = get_clauses_length_arr_b_b(clauses);
    int *clauses_length_arr_device;
    cudaMalloc(&clauses_length_arr_device, clauses.size() * sizeof(int));

    // Copy values to device
    cudaMemcpy(clauses_length_arr_device, clauses_length_arr_host, clauses.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate memory for an array that stores the result of each assignment in an array of length 2^nvars
    int *var_assignment_output_host = (int *) malloc(sizeof(int) * num_of_assignments);
    int *var_assignment_output_device;
    cudaMalloc(&var_assignment_output_device, sizeof(int) * num_of_assignments);

    // Spawn enough kernels to test all 2^nvars possible assignments
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_of_assignments) / threadsPerBlock + 1;
    // Call the kernel
    brute_force_kernel<<<blocksPerGrid, threadsPerBlock>>>(clauses_arr_device, clauses_length_arr_device, clauses.size(), var_assignment_output_device, num_of_assignments);

    // Synchronize threads
    cudaDeviceSynchronize();

    // Move values back to host code
    cudaMemcpy(var_assignment_output_host, var_assignment_output_device, sizeof(int) * num_of_assignments, cudaMemcpyDeviceToHost);

    int *iter;
    // Perform find_if to determine if there is a one
    iter = thrust::find_if(thrust::host, var_assignment_output_host, var_assignment_output_host + num_of_assignments, is_one());
    int result;
    if(iter == var_assignment_output_host + num_of_assignments){
        result = 0;
    }else{
        result = 1;
    }

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