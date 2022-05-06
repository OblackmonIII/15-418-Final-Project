#ifndef __MAIN_H__
#define __MAIN_H__

#include <stdio.h>
#include <vector>
#include <set>

class BruteForce {

    public:

        int brute_force_parallel(std::vector<std::set<int> > clauses, int nvars);
};

#endif