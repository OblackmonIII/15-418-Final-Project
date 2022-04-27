#ifndef __MAIN_H__
#define __MAIN_H__

#include <stdio.h>
#include <vector>
#include <set>

class BruteForce {

    public:

        int brute_force_parallel(std::vector<std::set<int> > clauses, int nvars);
        int **clauses_to_array(std::vector<std::set<int> > clauses);
        int *get_clauses_length_arr(std::vector<std::set<int> > clauses);
};

#endif