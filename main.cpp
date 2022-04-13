#include <iostream>
#include <set>
#include <iterator>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>

using namespace std;

bool test_assignment(bool *var_assignment, int nvars, std::vector<set<int>> clauses, int nclauses){

    // Iterate over clauses
    // Assign variables found in clauses based on array
    // If the clause returns false, return 0
    // Otherwise continue to the next clause
    // If all clauses return true, return true

    bool result = false;
    for(int i = 0; i < nclauses; i++){
        
        // Get iterator for the set
        //itr = clauses[i].begin();
       
        for(set<int>::iterator itr = clauses[i].begin(); itr != clauses[i].end(); itr++){
            // Take absolute value of the variable
            // if the value is false, negate the assignment
            // otw, leave it the same             
            int index = std::abs(*itr) - 1;
            result = result || var_assignment[index];
        } 
        
        if(!result){
        
            return result;
        }

        result = false;
    }

    return result;
}

std::string clauses_to_string(std::vector<set<int> > clauses, int nclauses){

    std::string result = "";
    for(int i = 0; i < nclauses; i++){

        set<int>::iterator itr;
        for(itr = clauses[i].begin(); itr != clauses[i].end(); itr++){
            // Take absolute value of the variable
            // if the value is false, negate the assignment
            // otw, leave it the same             

            result += to_string(*itr) + " | ";
        } 
       
        result.resize(result.size() - 3);
        
        result += " & "; 
    }
    
    result.resize(result.size() - 3);
    result += "\n";

    return result;
}

std::vector<std::string> get_clause_literal_vector(std::string clause_string){

    string space_delimiter = " ";
    vector<std::string> literals{};

    size_t pos = 0;
    while ((pos = clause_string.find(space_delimiter)) != string::npos) {
        literals.push_back(clause_string.substr(0, pos));
        clause_string.erase(0, pos + space_delimiter.length());
    }
    
    literals.push_back(clause_string.substr(0, clause_string.size() - 2));

    return literals;
}

std::set<int> parse_line_to_variables(std::string clause_string){

    std::vector<string> clause_literal_vector = get_clause_literal_vector(clause_string);
    std::set<int> var_set;

    for(int i = 0; i < clause_literal_vector.size(); i++){
       
       var_set.insert(std::stoi(clause_literal_vector[i])); 
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

    // Open file
    std::ifstream input_file;
    input_file.open(argv[1], std::ifstream::in);
    if(input_file.is_open()){
        
        initialize_variables(&input_file, &nvars, &clauses, &nclauses);
    }else{
        
        return -1;
    }

    //cin >> nvars >> nclauses;
    bool *var_assignments = new bool[nvars]; 

    // Construct SAT struct
    /*
    SAT Struct ideas
    - Store two items: an array of assignments and the formula. The array will be booleans
    and the formula an array of sets of variables. Restrict this to conjunctive normal form.
    */


    std::string clauses_string = clauses_to_string(clauses, nclauses);
    cout << clauses_string;

    // Call DPLL on the struct
    /*
    - Implement serial result
    */
    // Print result
    return 0;
}
