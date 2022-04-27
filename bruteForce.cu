#include "bruteForce.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

__device__ int get_nth_bit(int value, int n){

    return (value >> n) & (1);
}

__device__ bool test_assignment(int var_assignment_val, int **clauses, int *clause_length_arr, int nclauses){

    // Iterate over clauses
    // Assign variables found in clauses based on array
    // If the clause returns false, return 0
    // Otherwise continue to the next clause
    // If all clauses return true, return true

    bool result = false;
    bool var_assignment;
    for(int i = 0; i < nclauses; i++){
        
        result = false;

        for(int j = 0; j < clause_length_arr[i]; j++){
            // Take absolute value of the variable
            // if the value is false, negate the assignment
            // otw, leave it the same             
            int index = std::abs(clauses[i][j]) - 1;
            var_assignment = get_nth_bit(var_assignment_val, index) == 0 ? false : true;

            if(clauses[i][j] > 0){

                result = result || var_assignment;
            }else{

                result = result || !var_assignment;
            }
        } 
        
        if(!result){
        
            return result;
        }

    }

    return result;
}

__global__ void brute_force_kernel(int **clauses_arr_device, int *clauses_length_arr_device, int nclauses, int *var_assignment_output_device, int num_of_assignments){

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
    var_assignment_output_device[threadId] = test_assignment(threadId, clauses_arr_device, clauses_length_arr_device, nclauses);

}

struct is_one
{
  __host__ __device__
  bool operator()(int x)
  {
    return x == 1;
  }
};

__host__ int **BruteForce::clauses_to_array(std::vector<std::set<int> > clauses){

    int **clauses_arr = (int **) malloc(clauses.size() * sizeof(int *));

    for(int i = 0; i < clauses.size(); i++){
        
        clauses_arr[i] = (int *) malloc(clauses[i].size() * sizeof(int));

        std::set<int>::iterator pure_literal_itr;
        int j = 0;
        for(pure_literal_itr = clauses[i].begin(); pure_literal_itr != clauses[i].end(); pure_literal_itr++){

            clauses_arr[i][j] = *pure_literal_itr;
            j++;
        }
    }


    return clauses_arr;
}

int *BruteForce::get_clauses_length_arr(std::vector<std::set<int> > clauses){

    int *clauses_length_arr = (int *) malloc(clauses.size() * sizeof(int));

    for(int i = 0; i < clauses.size(); i++){

        clauses_length_arr[i] = clauses[i].size();
    }

    return clauses_length_arr;
}

int BruteForce::brute_force_parallel(std::vector<std::set<int> > clauses, int nvars){

    // Number of assignments
    int num_of_assignments = (double) pow(2.0, (double) nvars);

    // Allocate memory for the clauses on the host, fill with clause arrays, and move to the device
    int **clauses_arr_host = clauses_to_array(clauses);
    int **clauses_arr_device;
    cudaMalloc(&clauses_arr_device, clauses.size() * sizeof(int *));
    for(int i = 0; i < clauses.size(); i++){

        cudaMalloc(&(clauses_arr_device[i]), clauses[i].size() * sizeof(int));

        // Copy values to device
        cudaMemcpy(clauses_arr_device[i], clauses_arr_host[i], clauses[i].size() * sizeof(int), cudaMemcpyHostToDevice);
    }


    // Allocate memory that maps clause index to the number of variables in the clause
    int *clauses_length_arr_host = get_clauses_length_arr(clauses);
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

    // Scan through array to determine if there is a one
    thrust::exclusive_scan(thrust::host, var_assignment_output_host, var_assignment_output_host + num_of_assignments, var_assignment_output_host, 0);

    int *iter;
    // Perform find_if to determine if there is a one
    iter = thrust::find_if(thrust::device, var_assignment_output_host, var_assignment_output_host + num_of_assignments, is_one());
    int result = *iter;
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