#include "mixed_dpll.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <iterator>
#include <set>

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

__device__ bool contains_literal(int clauseID, int literal, int *clauses_arr_device, int *clauses_start_arr_device, int *clauses_length_arr_device){

    /* Loop through array segment and if it contains literal, then return true. Otherwise return false.*/

    for(int i = clauses_start_arr_device[clauseID]; i < clauses_start_arr_device[clauseID] + clauses_length_arr_device[clauseID]; i++){

        if(clauses_arr_device[i] == literal){

            return true;
        }
    }

    return false;
}

__device__ void remove_clause(int clauseID, int *clauses_arr_device, int *clauses_start_arr_device, int *clauses_length_arr_device){

    /* Removing a clause simply means zeroing out variable values  and saying that the length of the clause is -1*/

    for(int i = clauses_start_arr_device[clauseID]; i < clauses_start_arr_device[clauseID] + clauses_length_arr_device[clauseID]; i++){

        clauses_arr_device[i] = 0;
    }

    clauses_start_arr_device[clauseID] = -1;

}

__device__ void remove_literal_from_clause(int clauseID, int literal, int *clauses_arr_device, int *clauses_start_arr_device, int *clauses_length_arr_device){

    /* Removing a literal from a clause simply menas zeroing out the location the literal appears in the clause*/

    for(int i = clauses_start_arr_device[clauseID]; i < clauses_start_arr_device[clauseID] + clauses_length_arr_device[clauseID]; i++){

        if(clauses_arr_device[i] == literal){

            clauses_arr_device[i] = 0;
        }
    }
}

__global__ void unit_propagate_kernel(int literal, int *clauses_arr_device, int *clauses_start_arr_device, int *clauses_length_arr_device, int nclauses){

    /*
        Each thread is assigned a clause and will eliminate unit clauses if they exist. Eliminating a unit
        clause simply means finding clauses with only one nonzero value and then changing it to a zero.
    */
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if(clauses_start_arr_device[threadId] != -1 && clauses_length_arr_device[threadId] != 0 && threadId < nclauses){

        if(contains_literal(threadId, literal, clauses_arr_device, clauses_start_arr_device, clauses_length_arr_device)){

            remove_clause(threadId, clauses_arr_device, clauses_start_arr_device, clauses_length_arr_device);
        }

        remove_literal_from_clause(threadId, -1 * literal, clauses_arr_device, clauses_start_arr_device, clauses_length_arr_device);
    }

}

__global__ void pure_literal_kernel(int literal, int *clauses_arr_device, int *clauses_start_arr_device, int *clauses_length_arr_device, int nclauses){

    /*
        Each thread is assigned a clause and will the clause if it contains a pure literal. Eliminating a
        clause that contains a pure literal simply means finding clauses with the literal in question and zeroing out the variables in it.
    */
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if(clauses_start_arr_device[threadId] != -1 && clauses_length_arr_device[threadId] != 0){

        if(contains_literal(threadId, literal, clauses_arr_device, clauses_start_arr_device, clauses_length_arr_device)){

            remove_clause(threadId, clauses_arr_device, clauses_start_arr_device, clauses_length_arr_device);
        }
    }

}

struct is_one
{
  __host__ __device__
  bool operator()(int x)
  {
    return x == 1;
  }
};

__host__ int *clauses_to_array(std::vector<std::set<int> > clauses, int sum_length_of_clauses){

    int *clauses_arr = (int *) malloc(sum_length_of_clauses * sizeof(int));
    int clause_offset = 0;
    int j = 0;

    for(int i = 0; i < clauses.size(); i++){
        
        j = 0;
        std::set<int>::iterator pure_literal_itr;
        for(pure_literal_itr = clauses[i].begin(); pure_literal_itr != clauses[i].end(); pure_literal_itr++){

            clauses_arr[j + clause_offset] = *pure_literal_itr;
            j++;
        }

        clause_offset += clauses[i].size();
    }


    return clauses_arr;
}

__host__ int *get_clauses_length_arr(std::vector<std::set<int> > clauses){

    int *clauses_length_arr = (int *) malloc(clauses.size() * sizeof(int));

    for(int i = 0; i < clauses.size(); i++){

        clauses_length_arr[i] = clauses[i].size();
    }

    return clauses_length_arr;
}

__host__ int *get_clauses_start_arr(std::vector<std::set<int> > clauses){

    int *clauses_start_arr = (int *) malloc(clauses.size() * sizeof(int));
    clauses_start_arr[0] = 0;

    for(int i = 1; i < clauses.size(); i++){

        clauses_start_arr[i] = clauses_start_arr[i - 1] + clauses[i - 1].size();
    }

    return clauses_start_arr;
}


int get_sum_length_of_clauses(std::vector<std::set<int> > clauses){

    int sum = 0;

    for(int i = 0; i < clauses.size(); i++){

        sum += clauses[i].size();
    }


    return sum;
}

int unit_clauses_present(int *clauses_arr_host, int sum_length_of_clauses, int *clauses_length_arr_host, int *clauses_start_arr_host, int num_of_clauses){

    // Return 0 if no unit clauses are present
    // Return l if a unit clause is present and l is the variable that composes the unit clause

    int literal = 0;

    for(int i = 0; i < num_of_clauses; i++){

        if(clauses_start_arr_host[i] != -1 && clauses_length_arr_host[i] != 0){

            int var_count = 0;
            for(int j = clauses_start_arr_host[i]; j < clauses_start_arr_host[i] + clauses_length_arr_host[i]; j++){

                if(clauses_arr_host[j] != 0){

                    var_count += 1;
                    literal = clauses_arr_host[j];
                }

            }

            if(var_count == 1){

                return literal;
            }

            var_count = 0;
            literal = 0;
        }
    }

    return literal;
}

int pure_literal_present(int *clauses_arr_host, int sum_length_of_clauses, int *clauses_start_arr_host, int *clauses_length_arr_host, int num_of_clauses){

    // Return 0 if no pure literal is present
    // Return l if a pure literal present and l is the pure literal

    std::set<int> variable_set;
    for(int i = 0; i < num_of_clauses; i++){

        // Checks if we are looking at a valid clause
        if(clauses_start_arr_host[i] != -1 && clauses_length_arr_host[i] != 0){

            for(int j = clauses_start_arr_host[i]; j < clauses_start_arr_host[i] + clauses_length_arr_host[i]; j++){

                variable_set.insert(clauses_arr_host[i]);
            }
        }
    }

    std::set<int>::iterator it;
    std::set<int>::iterator neg_it;
    for(it = variable_set.begin(); it != variable_set.end(); it++){

        if(*it != 0){

            neg_it = variable_set.find(-1 * (*it));
            if(neg_it == variable_set.end()){

                return *it;
            }
        }
    }

    return 0;
}

int get_number_of_clauses(int *clauses_start_arr_host, int *clauses_length_arr_host, int num_of_clauses){

    // Count up the number of clauses that do not start at -1 and have a length greater than or equal to 0

    int count = 0;
    for(int i = 0; i < num_of_clauses; i++){

        if(clauses_start_arr_host[i] != -1){
            count ++;
        }
    }

    return count;
}

bool check_for_empty_clauses(int *clauses_start_arr_host, int *clauses_length_arr_host, int num_of_clauses){

    // Check if a clause is still in the expression, but has length 0

    for(int i = 0; i < num_of_clauses; i++){

        if(clauses_start_arr_host[i] != -1 && clauses_length_arr_host[i] == 0){

            return true;
        }
    }

    return false;

}

int get_literal(int *clauses_arr_host, int sum_length_of_clauses){

    for(int i = 0; i < sum_length_of_clauses; i++){

        if(clauses_arr_host[i] != 0){

            return clauses_arr_host[i];
        }
    }

    return 0;

}

void copy_array(int *src, int *dest, int length){

    // Copies length total values from src to dest. Length must be less than or equal to the src length and dest length

    for(int i = 0; i < length; i++){

        dest[i] = src[i];
    }
}

int MixedDPLL::mixed_dpll_parallel(int *clauses_arr_host, int sum_length_of_clauses, int *clauses_start_arr_host, int *clauses_length_arr_host, int num_of_clauses, int nvars){

    // Return 1 if satisfiable
    // Return 0 if unsatisfiable

    // Allocate CUDA memory for clauses
    int *clauses_arr_device;
    cudaMalloc(&clauses_arr_device, sum_length_of_clauses * sizeof(int));
    cudaMemcpy(clauses_arr_device, clauses_arr_host, sum_length_of_clauses * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate CUDA memory for clause length array
    int *clauses_length_arr_device;
    cudaMalloc(&clauses_length_arr_device, num_of_clauses * sizeof(int));
    cudaMemcpy(clauses_length_arr_device, clauses_length_arr_host, num_of_clauses * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate CUDA memory for clause start array
    // Spawn enough threads for one clause per thread
    int *clauses_start_arr_device;
    cudaMalloc(&clauses_start_arr_device, num_of_clauses * sizeof(int));
    cudaMemcpy(clauses_start_arr_device, clauses_start_arr_host, num_of_clauses * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_of_clauses) / threadsPerBlock + 1;
    int result = 0;

    // This code block handles unit propagation
    int literal = unit_clauses_present(clauses_arr_host, sum_length_of_clauses, clauses_length_arr_host, clauses_start_arr_host, num_of_clauses);
    while(literal != 0){

        unit_propagate_kernel<<<blocksPerGrid, threadsPerBlock>>>(literal, clauses_arr_device, clauses_start_arr_device, clauses_length_arr_device, num_of_clauses);
        cudaDeviceSynchronize();
        cudaMemcpy(clauses_arr_host, clauses_arr_device, sum_length_of_clauses * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(clauses_start_arr_host, clauses_start_arr_device, num_of_clauses * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(clauses_length_arr_host, clauses_length_arr_device, num_of_clauses * sizeof(int), cudaMemcpyDeviceToHost);

        literal = unit_clauses_present(clauses_arr_host, sum_length_of_clauses, clauses_length_arr_host, clauses_start_arr_host, num_of_clauses);
    }

    // This code block handles pure literal assignmet
    literal = pure_literal_present(clauses_arr_host, sum_length_of_clauses, clauses_length_arr_host, clauses_start_arr_host, num_of_clauses);
    while(literal != 0){

        // Needs parallelism
        // If a pure literal is detected, the clauses can cleared of that literal in parallel
        // You could try passing a list of pure literals to each thread so that they can clear them in one swoop
        pure_literal_kernel<<<blocksPerGrid, threadsPerBlock>>>(literal, clauses_arr_device, clauses_start_arr_device, clauses_length_arr_device, num_of_clauses); 
        cudaDeviceSynchronize();
        cudaMemcpy(clauses_arr_host, clauses_arr_device, sum_length_of_clauses * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(clauses_start_arr_host, clauses_start_arr_device, num_of_clauses * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(clauses_length_arr_host, clauses_length_arr_device, num_of_clauses * sizeof(int), cudaMemcpyDeviceToHost);

        literal = pure_literal_present(clauses_arr_host, sum_length_of_clauses, clauses_length_arr_host, clauses_start_arr_host, num_of_clauses);

    }

    // Check if there are no more clauses
    if(get_number_of_clauses(clauses_start_arr_host, clauses_length_arr_host, num_of_clauses) == 0){

        return 1;
    }

    // Check for empty clauses
    if(check_for_empty_clauses(clauses_start_arr_host, clauses_length_arr_host, num_of_clauses)){

        // Checking for empty clauses can probably be done in parallel
        return 0;
    }

    // Need to change this block to match the datatypes we are working with
    literal = get_literal(clauses_arr_host, sum_length_of_clauses);

    int *new_clauses_arr_host = (int *) malloc((sum_length_of_clauses + 1) * sizeof(int));
    int *new_clauses_start_arr_host = (int *) malloc((num_of_clauses + 1) * sizeof(int));
    int *new_clauses_length_arr_host = (int *) malloc((num_of_clauses + 1) * sizeof(int));

    // Copy the array values over and then append new values to the end
    copy_array(clauses_arr_host, new_clauses_arr_host, sum_length_of_clauses);
    copy_array(clauses_start_arr_host, new_clauses_arr_host, num_of_clauses);
    copy_array(clauses_length_arr_host, new_clauses_arr_host, num_of_clauses); 

    new_clauses_arr_host[sum_length_of_clauses] = literal;
    new_clauses_length_arr_host[num_of_clauses] = 1;
    new_clauses_start_arr_host[num_of_clauses] = sum_length_of_clauses;

    result = mixed_dpll_parallel(new_clauses_arr_host, sum_length_of_clauses + 1, new_clauses_start_arr_host, new_clauses_length_arr_host, num_of_clauses + 1, nvars);

    if(result == 0){

        new_clauses_arr_host[sum_length_of_clauses] = -1 * literal;
        result = mixed_dpll_parallel(new_clauses_arr_host, sum_length_of_clauses + 1, new_clauses_start_arr_host, new_clauses_length_arr_host, num_of_clauses + 1, nvars);
    }

    free(new_clauses_arr_host);
    free(new_clauses_start_arr_host);
    free(new_clauses_length_arr_host);

    cudaFree(clauses_arr_device);
    cudaFree(clauses_start_arr_device);
    cudaFree(clauses_length_arr_device);

    return result;
}

int MixedDPLL::dpll_parallel_wrapper(std::vector<std::set<int> > clauses, int nvars){

    // Allocate memory for the clauses on the host, fill with clause arrays, and move to the device
    int sum_length_of_clauses = get_sum_length_of_clauses(clauses);
    int *clauses_arr_host = clauses_to_array(clauses, sum_length_of_clauses);
    int *clauses_length_arr_host = get_clauses_length_arr(clauses);
    int *clauses_start_arr_host = get_clauses_start_arr(clauses);

    int result = mixed_dpll_parallel(clauses_arr_host, sum_length_of_clauses, clauses_length_arr_host, clauses_start_arr_host, clauses.size(), nvars);
    
    free(clauses_arr_host);
    free(clauses_length_arr_host);
    free(clauses_start_arr_host);

    return result;
}