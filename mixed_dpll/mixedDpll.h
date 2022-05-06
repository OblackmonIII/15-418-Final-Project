#ifndef __MAIN_H__
#define __MAIN_H__

#include <stdio.h>
#include <vector>
#include <set>
#include <iterator>

class MixedDPLL {

    public:

        int dpll_parallel_wrapper(std::vector<std::set<int> > clauses, int nvars);
        int mixed_dpll_parallel(int *clauses_arr_host, int sum_length_of_clauses, int *clauses_start_arr_host, int *clauses_length_arr_host, int num_of_clauses, int nvars);
};

#endif