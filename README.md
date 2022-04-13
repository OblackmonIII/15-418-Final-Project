# **Parallel SAT Solver using CUDA - Odell Blackmon III & Viraj Puri**

## **Summary**
We are going to implement a parallelized version of the DPLL algorithm for solving SAT problems. We plan to accomplish this using the NVIDIA CUDA GPUs. Using CUDA, we intend to show how/if parallelization of the algorithm would speed up computation in this case, whether it scales well/to what extent as the number of threads/cores increases, as well as how this compares to a serial implementation. This will be conveyed through visual representations such as graphs to display core speedup, as well as the effect of minor parameter tweakage such as number of blocks, dimensions, etc. 

## **Background**
The boolean satisfiability problem or SAT problem is a popular problem of determining if a boolean expression has an assignment of values to variables that will make the expression true. The problem is trivial when it comes to expressions with a small number of variables. However, the search space for possible solutions quickly balloons as the number of variables increases. The SAT problem is known to be NP-Complete which means no known polynomial time algorithm exists as well as this reduces to other NP-Complete problems. By solving a SAT problem or getting a close approximation for a SAT problem, a solution can be transformed into a similar solution or approximation for other NP-Complete problems.

As mentioned above, the SAT Problem is NP-Complete, but there are still algorithms that can be used to come up with solutions. For instance, the Davis-Putnam-Logemann-Loveland algorithm is a backtracking based algorithm for finding a satisfiable assignment of variables if one exists. The worst case runtime of it is exponential, but this run time can be improved through parallelism. The search space for potential solutions can be split across parallel operating units. Thus, there are indeed different axes of parallelism, and not only do we seek to exploit such axes, but we also aim to use CUDA to figure out to what degree of improvement does speedup top out at in comparison to the serial version.

## **The Challenge**
The problem is challenging for a number of reasons. To begin, the search space can be exponential proportional to the number of variables. Even with a significant number of threads, key insights into parallelism will be needed to solve this problem. Secondly, synchronization across this many threads will be difficult to reason over since it is unclear at this moment how work will be divided amongst threads as well as where the overlaps in the work will appear. These are only some of the challenges that this project faces.

## **Resources**
First and foremost, the primary development resource we will be using are the NVIDIA GPUs from the GHC machines. This is for one of several reasons, the primary being, not only are we familiar with/comfortable with development on these remote GPUs through the work that has been done so far in 15-418 (especially during Assignment 2 which focused primarily on CUDA development using the GHC machines), but also because the GPUs found on these machines (NVIDIA RTX 2080) is not only native to NVIDIA’s CUDA dev kit, but also is much faster than the integrated Intel GPU found in our local Macbook Pro laptops. Thus, the GHC machines would be ideal for development, as well as for scaling to a large number of cores/threads to display the sheer scaling benefits from our results. Additionally, we’re going to use the pseudocode for the DPLL algorithm found from CMU’s 15-414 course on bug catching, by Andrė Platzer, and Matt Fredrikson at https://www.cs.cmu.edu/~15414/f17/lectures/10-dpll.pdf for our serial version to be used as a benchmark, as well as a baseline reference to implement our parallel version in CUDA.

## **Goals and Deliverables**
* 75%
  * A sequential SAT solver
  * A sequential SAT solver with adjustable time out
* 100%
  * A SAT solver that runs on GPUs
  * Speedup graphs for different block sizes and thread counts
* 125%
  * An interactive SAT solver where a user can input a boolean expression and the program will output a solution if the expression is feasible
  * The same interactive program will also output how many possible assignments exist, how many assignments were checked, and how long it took
  * Output the tree that was constructed when trying different variable values to solve the SAT expression 

![alt text](https://upload.wikimedia.org/wikipedia/commons/d/dc/Dpll11.png "Title")


## **Platform Choice**
Our platforms of choice are the C++ language, the CUDA toolkit, and the NVIDIA GPUs found in the GHC machines. The reasoning for these choices are multiple, starting with the basic fact that both team members in our group are most comfortable with CUDA among the various different choices of parallel toolkits available to us (such as OpenMP, MPI, etc). Additionally, C++ is not only the language of choice for the vast majority of projects/assignments in this course, but also is the primary language used in CUDA development, so it was only a natural choice. Furthermore, this makes debugging, as well as implementing any stretch goals a much less tedious task, due to the vast majority of documentation available. C++ also happens to be one of the fastest languages available, second only to C, just being a bit safer, making it ideal for not only showing pure speedup, but also having a balance of ease of implementation. Finally, both members in our group also thought it would be interesting to use GPU resources as a means to implement something that users would not think traditionally would benefit from GPU resources, in contrast to an example such as rendering an image, or graphic manipulation. 
## **Input File Format**
This program will only handle boolean expressions in conjunctive normal form.
The first line of the input file should have two integers. The first integer will denote the number of variables and the second one should denote the number of clauses.

The next lines should define the clauses. Each clause will be a disjunction of variables ranging from 1 to the number of variables defined on the previous line. Each variable can either be positive or negtative to denote whether a variable is a literal or negated respectively.
## **MILESTONE**
So far, we have implemented a working version of a baseline serial implementation of the DPLL algorithm, created test cases for our serial/parallel implementations, and almost done identifying some axes of parallelism, and also began to implement our CUDA-improved parallel version. Firstly, with our DPLL serial implementation, this was not extremely challenging as we have seen the algorithm before, plus, there are many referential high level descriptions of the algorithm, so implementing this was somewhat straightforward, we just ran into some (fairly normal) compile-time bugs, as well as some general logical issues. We were able to solve this by then dedicating some time to actually creating some test cases, which proved to be a bit challenging, as we had to find a representation for SAT problems that we could reliably use in C++, but through some grind, we used these test cases to get the serial version working so far, and will use these test cases to test overall correctness of our parallel version as well. We will spend time later in the semester to create some very large test cases such that we can see the benefits of it when scaled heavily using CUDA. 

We also realized the parallelizing using CUDA for DPLL for SAT is actually quite a more daunting task than we had thought, as we inspected the algorithm’s high-level description to identify the dependencies/loops, and found some initial challenges here, as the DPLL algorithm did not make use of for loops similar to the examples from class/the online NVIDIA Cuda examples, rather, focused more so on using recursion, white loops, and if statements/branches for the algorithm, which requires further work, since in CUDA, all blocks are independent of each other, so we needed to find out where dependencies did not lie. We also realized that GPU’s tend to have much more limited memory than CPUs so the parallelism/recursion stack would be very difficult to implement on large test cases which is what we intend to use to show scalability benefits, so that is currently our main task while we simultaneously work to implement the parallel version of this algorithm, as well as deciding ideal block sizes, etc. later. 

So, far we are doing decently well with our calendar, only thing is we are about a few days behind schedule in that we would have liked to get our axes of parallelism solidified by about this time, but this proved to be a bit harder than we anticipated since the recursion/loops posed a bit more of a challenge than thought, so now that we have both members of our teammates finished a good deal of work for our other classes, we intend to focus on catching up ground to quickly get back on track by this coming weekend, as in the immediate future we intend to first focus on preparing for and completing the upcoming midterm examination, then immediately focusing all our efforts to getting this done. Overall, this isn’t that bad of a roadblock, just requires some extra time to dedicate toward this/identifying any outside support if we need it.

At poster session time, we intend to show various graphs as our main visual element. During the poster session time, we intend to use graphs/trend patterns to show (the potential) benefits of using parallelization with CUDA to speed up the DPLL algorithm, and how this scales on more cores/threads compared to either a baseline serial implementation or a single thread/core implementation (to be determined later). With this we can show overall speedup and the kinds of performance gains we get, which will also allow us to talk about any bottlenecks that may be encountered for future analysis as well as any limitations of our analysis.

At the present time we do not have preliminary results as we only have our serial version, but do not yet have large enough test cases developed to show any serious slowdown of the serial version since creating SAT type test cases from scratch have long proved to be a bit harder than anticipated.

At present, our current concerns are just figuring out how to get around the recursion using CUDA/axes of parallelism and finish implementing the parallel version of the code, but this is just a matter of just coding and doing the work to complete this which for the second half of this project timeline we intend to dedicate much more time to to complete and finish, especially as the rest of our classes wrap up.


## **Schedule**
| Week | Items |
| ---- | ----- |
| Week 1 (Mar 21 - Mar 25) | Finalize project idea/proposal, get environment/Git setup. |
| Week 2 (Mar 28 - Apr 1) | Implement serial version, establish benchmark for serial version to improve on, compile baseline visualization, identify potential axes of parallelism. |
| Week 3 (Apr 4 - Apr 8)| Begin to implement parallel version, work on/finish milestone report |
| Week 4 (Apr 11 - Apr 15) MIDPOINT DUE | Finish implementing parallel version, tweak where necessary (such as number of threads per block) to identify best possible parallel version, begin work on visualizing results
| Week 5 (Apr 18 - Apr 22) | Scale results up to a larger number of cores to identify where speedup gives diminishing returns on our machine, compile final graphs, as well as optimize code where necessary to identify best performance. |
| Week 6 (Apr 25 - Apr 29) | Finalize results, finish final report & submit, clean up code. |
| Week 7 (May 2 - May 6) | Work on final poster, ready/practice for 5 minute presentation. |
