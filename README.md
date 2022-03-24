# **Parallel SAT Solver using CUDA - Odell Blackmon III & Viraj Puri**

## **Summary**
We are going to implement a parallelized version of the DPLL algorithm for solving SAT problems. We plan to accomplish this using the NVIDIA CUDA GPUs. Using CUDA, we intend to show how/if parallelization of the algorithm would speed up computation in this case, whether it scales well/to what extent as the number of threads/cores increases, as well as how this compares to a serial implementation. This will be conveyed through visual representations such as graphs to display core speedup, as well as the effect of minor parameter tweakage such as number of blocks, dimensions, etc. 

## **Background**
The boolean satisfiability problem or SAT problem is a popular problem of determining if a boolean expression has an assignment of values to variables that will make the expression true. The problem is trivial when it comes to expressions with a small number of variables. However, the search space for possible solutions quickly balloons as the number of variables increases. The SAT problem is known to be NP-Complete which means no known polynomial time algorithm exists as well as this reduces to other NP-Complete problems. By solving a SAT problem or getting a close approximation for a SAT problem, a solution can be transformed into a similar solution or approximation for other NP-Complete problems.

As mentioned above, the SAT Problem is NP-Complete, but there are still algorithms that can be used to come up with solutions. For instance, the Davis-Putnam-Logemann-Loveland algorithm is a backtracking based algorithm for finding a satisfiable assignment of variables if one exists. The worst case runtime of it is exponential, but this run time can be improved through parallelism. The search space for potential solutions can be split across parallel operating units. Thus, there are indeed different axes of parallelism, and not only do we seek to exploit such axes, but we also aim to use CUDA to figure out to what degree of improvement does speedup top out at in comparison to the serial version.

# **The Challenge**
The problem is challenging for a number of reasons. To begin, the search space can be exponential proportional to the number of variables. Even with a significant number of threads, key insights into parallelism will be needed to solve this problem. Secondly, synchronization across this many threads will be difficult to reason over since it is unclear at this moment how work will be divided amongst threads as well as where the overlaps in the work will appear. These are only some of the challenges that this project faces.

## **Resources**
First and foremost, the primary development resource we will be using are the NVIDIA GPUs from the GHC machines. This is for one of several reasons, the primary being, not only are we familiar with/comfortable with development on these remote GPUs through the work that has been done so far in 15-418 (especially during Assignment 2 which focused primarily on CUDA development using the GHC machines), but also because the GPUs found on these machines (NVIDIA RTX 2080) is not only native to NVIDIA’s CUDA dev kit, but also is much faster than the integrated Intel GPU found in our local Macbook Pro laptops. Thus, the GHC machines would be ideal for development, as well as for scaling to a large number of cores/threads to display the sheer scaling benefits from our results. Additionally, we’re going to use the pseudocode for the DPLL algorithm found from CMU’s 15-414 course on bug catching, by Andrė Platzer, and Matt Fredrikson at https://www.cs.cmu.edu/~15414/f17/lectures/10-dpll.pdf for our serial version to be used as a benchmark, as well as a baseline reference to implement our parallel version in CUDA.

## **Goals and Deliverables**
75%
A sequential SAT solver
A sequential SAT solver with adjustable time out
100%
A SAT solver that runs on GPUs
Speedup graphs for different block sizes and thread counts
125%
An interactive SAT solver where a user can input a boolean expression and the program will output a solution if the expression is feasible
The same interactive program will also output how many possible assignments exist, how many assignments were checked, and how long it took
Output the tree that was constructed when trying different variable values to solve the SAT expression 

## **Platform Choice**
Our platforms of choice are the C++ language, the CUDA toolkit, and the NVIDIA GPUs found in the GHC machines. The reasoning for these choices are multiple, starting with the basic fact that both team members in our group are most comfortable with CUDA among the various different choices of parallel toolkits available to us (such as OpenMP, MPI, etc). Additionally, C++ is not only the language of choice for the vast majority of projects/assignments in this course, but also is the primary language used in CUDA development, so it was only a natural choice. Furthermore, this makes debugging, as well as implementing any stretch goals a much less tedious task, due to the vast majority of documentation available. C++ also happens to be one of the fastest languages available, second only to C, just being a bit safer, making it ideal for not only showing pure speedup, but also having a balance of ease of implementation. Finally, both members in our group also thought it would be interesting to use GPU resources as a means to implement something that users would not think traditionally would benefit from GPU resources, in contrast to an example such as rendering an image, or graphic manipulation. 

## **Schedule**
Week
Items
Week 1 (Mar 21 - Mar 25)
Finalize project idea/proposal, get environment/Git setup.
Week 2 (Mar 28 - Apr 1)
Implement serial version, establish benchmark for serial version to improve on, compile baseline visualization, identify potential axes of parallelism
Week 3 (Apr 4 - Apr 8) 
Begin to implement parallel version, work on/finish milestone report
Week 4 (Apr 11 - Apr 15) MIDPOINT DUE
Finish implementing parallel version, tweak where necessary (such as number of threads per block) to identify best possible parallel version, begin work on visualizing results
Week 5 (Apr 18 - Apr 22)
Scale results up to a larger number of cores to identify where speedup gives diminishing returns on our machine, compile final graphs, as well as optimize code where necessary to identify best performance.
Week 6 (Apr 25 - Apr 29)
Finalize results, finish final report & submit, clean up code.
Week 7 (May 2 - May 6)
Work on final poster, ready/practice for 5 minute presentation.
