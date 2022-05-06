#include <iostream>
#include <set>
#include <iterator>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include "brute_force/bruteForce.h"
#include "CycleTimer.h"

using namespace std;


bool test_assignment(bool *var_assignment, int nvars, std::vector<set<int> > clauses, int nclauses){

    // Iterate over clauses
    // Assign variables found in clauses based on array
    // If the clause returns false, return 0
    // Otherwise continue to the next clause
    // If all clauses return true, return true

    bool result = false;
    for(int i = 0; i < nclauses; i++){
        
        result = false;

        for(set<int>::iterator itr = clauses[i].begin(); itr != clauses[i].end(); itr++){
            // Take absolute value of the variable
            // if the value is false, negate the assignment
            // otw, leave it the same             
            int index = std::abs(*itr) - 1;

            if(*itr > 0){

                result = result || var_assignment[index];
            }else{

                result = result || !var_assignment[index];
            }
        } 
        
        if(!result){
        
            return result;
        }

    }

    return result;
}

std::string clauses_to_string(std::vector<set<int> > clauses){

    if(clauses.size() == 0){
        return "";
    }
    std::string result = "";
    for(int i = 0; i < clauses.size(); i++){

        if(clauses[i].size() != 0){

            set<int>::iterator itr;
            for(itr = clauses[i].begin(); itr != clauses[i].end(); itr++){
                // Take absolute value of the variable
                // if the value is false, negate the assignment
                // otw, leave it the same             
                result += to_string(*itr) + " | ";
            } 
        
            result.resize(result.size() - 3);

            if(i != clauses.size() - 1){
                result += " & "; 
            }            
        }
    }

    result += "\n";

    return result;
}

std::vector<std::string> get_clause_literal_vector(std::string clause_string){

    std::istringstream iss(clause_string);   
    std::vector<std::string> result;

    string n;
    while(iss >> n)
    {
        result.push_back(n);
    }

    return result;
}

std::set<int> parse_line_to_variables(std::string clause_string){

    std::set<int> var_set;

    std::istringstream iss(clause_string);   

    int n;
    while(iss >> n)
    {
        var_set.insert(n);
    }

    return var_set;
}

void initialize_variables(ifstream *input_file, int *nvars, std::vector<set<int> > *clauses, int *nclauses){


    // Get nvars and nclauses
    string line;
    std::getline(*input_file, line);    
    std::vector<std::string> input_vars = get_clause_literal_vector(line);

    *nvars = std::stoi(input_vars[0]);
    *nclauses = std::stoi(input_vars[1]);
    
    // Fill in clauses
    for(int i = 0; i < *nclauses; i++){
       
        getline(*input_file, line);
        (*clauses).push_back(parse_line_to_variables(line));
    } 
    
}

bool unit_clauses_present(std::vector<set<int> > clauses){

    for(int i = 0; i < clauses.size(); i++){

        if(clauses[i].size() == 1){

            return true;
        }
    }


    return false;
}

int find_unit_clause(std::vector<set<int> > clauses){

    int unit_clause_variable = 0;

    for(int i = 0; i < clauses.size(); i++){

        if(clauses[i].size() == 1){

            unit_clause_variable = *clauses[i].begin();
            return unit_clause_variable;
        }
    }


    return unit_clause_variable;

}

std::vector<set<int> > unit_propagate(std::vector<set<int> > clauses, int nclauses, int unit_clause_variable){

    // Iterate over the clauses. Remove clauses that contain unit_clause_variable. Remove -1 * unit_clause_variable
    // from clauses

    std::vector<set<int> > new_clauses;
    set<int>::iterator negation_itr;

    for(int i = 0; i < nclauses; i++){

        if(clauses[i].find(unit_clause_variable) == clauses[i].end()){
            // If the clause does not contain the unit variable, then include in new_clauses without the negated variable

            negation_itr = clauses[i].find(-1 * unit_clause_variable);
            if(negation_itr == clauses[i].end()){
                // Did not find negation so we can add clauses as is
                new_clauses.push_back(clauses[i]);
            }else{
                // Did find negation, so remove negation then add clause
                clauses[i].erase(negation_itr);
                new_clauses.push_back(clauses[i]);
            }

        }else{
            // This is the case where the clause does contain the unit variable, so we ignore
            continue;
        }
    }

    return new_clauses;    
}


bool pure_literals_present(std::vector<set<int> > clauses, int nvars){

    bool literal_flag = false;
    bool neg_literal_flag = false;

    for(int i = 1; i <= nvars; i++){
        // Check if i and its negation show up in any clause
        // If not, it means it's pure and return true

        for(int j = 0; j < clauses.size(); j++){

            if(!literal_flag){

                literal_flag = clauses[j].find(i) != clauses[j].end();
            }

            if(!neg_literal_flag){

                neg_literal_flag = clauses[j].find(-1 * i) != clauses[j].end();
            }

        }

        if(literal_flag ^ neg_literal_flag){

            return true;

        }
    }

    return false;
}


int find_pure_literal(std::vector<set<int> > clauses, int nvars){

    int pure_literal_variable = 0;
    bool literal_flag = false;
    bool neg_literal_flag = false;

    for(int i = 1; i <= nvars; i++){
        // Check if i and its negation show up in any clause
        // If not, it means it's pure and return true

        for(int j = 0; j < clauses.size(); j++){

            if(!literal_flag){

                literal_flag = clauses[j].find(i) != clauses[j].end();
            }

            if(!neg_literal_flag){

                neg_literal_flag = clauses[j].find(-1 * i) != clauses[j].end();
            }

        }

        if(literal_flag ^ neg_literal_flag){

            if(literal_flag){

                pure_literal_variable = i;
            }else{

                pure_literal_variable = -1 * i;
            }

            return pure_literal_variable;

        }
    }

    return pure_literal_variable;
}

std::vector<set<int> > pure_literal_assign(std::vector<set<int> > clauses, int pure_literal_variable){

    std::vector<set<int> > new_clauses;
    set<int>::iterator pure_literal_itr;

    for(int i = 0; i < clauses.size(); i++){
        pure_literal_itr = clauses[i].find(pure_literal_variable);
        if(pure_literal_itr != clauses[i].end()){
            clauses[i].erase(pure_literal_itr);
        }else{

            new_clauses.push_back(clauses[i]);
        }

    }

    return new_clauses;
}
bool check_for_empty_clauses(std::vector<set<int> > clauses){

    for(int i = 0; i < clauses.size(); i++){

        if(clauses[i].empty()){

            return true;
        }
    }

    return false;
}

set<int> get_literals(std::vector<set<int> > clauses){

    set<int> literals;
    for(int i = 0; i < clauses.size(); i++){

        literals.insert(clauses[i].begin(), clauses[i].end());
    }

    return literals;
}

int dpll(std::vector<set<int> > clauses, int nvars){

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
    result = dpll(clauses, clauses.size());

    if(result == 0){
        clauses[clauses.size() - 1].erase(literal_choice);
        clauses[clauses.size() - 1].insert(-1 * literal_choice);
        result = dpll(clauses, clauses.size());
    }

    return result;
}


int **clauses_to_array(std::vector<set<int>> clauses){

    int **clauses_arr = (int **) malloc(clauses.size() * sizeof(int *));

    for(int i = 0; i < clauses.size(); i++){
        
        clauses_arr[i] = (int *) malloc(clauses[i].size() * sizeof(int));

        set<int>::iterator pure_literal_itr;
        int j = 0;
        for(pure_literal_itr = clauses[i].begin(); pure_literal_itr != clauses[i].end(); pure_literal_itr++){

            clauses_arr[i][j] = *pure_literal_itr;
            j++;
        }
    }


    return clauses_arr;
}

int *get_clauses_length_arr(std::vector<set<int>> clauses){

    int *clauses_length_arr = (int *) malloc(clauses.size() * sizeof(int));

    for(int i = 0; i < clauses.size(); i++){

        clauses_length_arr[i] = clauses[i].size();
    }

    return clauses_length_arr;
}


int main(int argc, char **argv){

    // Read input
    /*
    - Stick to conjunctive normal form
    - Have the user input the number of variables and the number of clauses
    - They will be able to enter each clause
    - Once all clauses are entered, then the program will begin
    */
    int nvars;
    int nclauses;
    std::vector<set<int> > clauses;

    string input_file_name;
    string mode;
    
    // Parsing commandline arguments
    if(argc < 3){

        cout << "Invalid number of arguments. Please try again by specifying an input file and computation mode" << endl;
        return -1;
    }

    for(int i = 1; i < argc; i++){

        if(string("-i").compare(argv[i]) == 0){

            input_file_name = argv[i + 1];
        }

        if(string("-m").compare(argv[i]) == 0){

            mode = argv[i + 1];
        }
    }

    // Open file
    std::ifstream input_file;
    input_file.open(input_file_name, std::ifstream::in);
    if(input_file.is_open()){
        
        initialize_variables(&input_file, &nvars, &clauses, &nclauses);
    }else{

        cout << "Failed to open file. Please ensure the file exists and the path is correct \n";
        return -1;
    }

    bool *var_assignments = new bool[nvars]; 

    // Construct SAT struct
    /*
    SAT Struct ideas
    - Store two items: an array of assignments and the formula. The array will be booleans
    and the formula an array of sets of variables. Restrict this to conjunctive normal form.
    */


    std::string clauses_string = clauses_to_string(clauses);
    cout << clauses_string;

    var_assignments[0] = false;
    var_assignments[1] = false;

    //cout << "Testing different assingments: " << test_assignment(var_assignments, nvars, clauses, nclauses) << endl;
    // Call DPLL on the struct
    int result;
    if(mode == "serial"){
        double startTime = CycleTimer::currentSeconds();
        result = dpll(clauses, nvars);
        double endTime = CycleTimer::currentSeconds();
        double serialDuration = endTime - startTime;
        printf("Overall serial time: %.6f s\t\n", serialDuration);
    }
    else if(mode == "brute_force"){
        double startTimeBrute = CycleTimer::currentSeconds();
        BruteForce *bf = new BruteForce();
        result = bf->brute_force_parallel(clauses, nvars);
        double endTimeBrute = CycleTimer::currentSeconds();
        double bruteDuration = endTimeBrute - startTimeBrute;
        printf("Overall brute force parallel time: %.6f s\t\n", bruteDuration);
    }

    // Print result
    cout << "1 if SAT 0 if not: " << result << endl;

    return 0;
}
