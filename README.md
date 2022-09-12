Effect of using different values of tolerance with barrier-free iterations in
[OpenMP]-based ordered levelwise [PageRank algorithm] for [link analysis].

**Unordered PageRank** is the *standard* approach of PageRank computation (as
described in the original paper by Larry Page et al. [(1)]), where *two*
*different rank vectors* are maintained; one representing the *current* ranks of
vertices, and the other representing the *previous* ranks. On the other hand,
**ordered PageRank** uses *a single rank vector*, representing the current ranks
of vertices [(2)]. This is similar to barrierless non-blocking implementations
of the PageRank algorithm by Hemalatha Eedi et al. [(3)]. As ranks are updated
in the same vector (with each iteration), the order in which vertices are
processed *affects* the final result (hence the adjective *ordered*). However,
as PageRank is an iteratively converging algorithm, results obtained with either
approach are *mostly the same*. **Barrier-free PageRank** is an *ordered*
*PageRank* where each thread processes a subset of vertices in the graph
independently, *without* waiting (with a barrier) for other threads to complete
an iteration. This minimizes unnecessary waits and allows each thread to be on a
*different iteration number* (which may or may not be beneficial for
convergence) [(3)]. **Monolithic PageRank** is the standard PageRank computation
where vertices are grouped by *strongly connected components* (SCCs). This
improves *locality* of memory accesses and thus improves performance [(4)].
**Levelwise PageRank** is a decomposed form of PageRank computation, where each
*SCC* is processed by *topological order* in the *block-graph* (all components
in a *level* are processed together *until* convergence, after which we proceed
to the next level). This decomposition allows for distributed computation
*without per-iteration communication*. However, it does not work on a graph
which includes *dead ends* (vertices with no outgoing edges, also called
dangling nodes) [(5)].

In this experiment, we perform two different approaches of barrier-free
iterations of *OpenMP-based ordered PageRank*; one in which each thread detects
convergence by measuring the difference between the previous and the current
ranks of all the vertices (**full**), and the other in which the difference is
measured between the previous and current ranks of only the subset of vertices
being processed by each thread (**part**). Both approahes are performed in
either the *default* way (with vertices arranged in vertex-id order), with the
*monolithic* approach (where vertices are grouped by SCCs), or with the
*levelwise* approach (where vertices are processed until convergence in groups
of levels in the block-graph). This is done while adjusting the tolerance `τ`
from `10^-1` to `10^-14` with `L∞-norm` as the tolerance function. We also
compare it with OpenMP-based unordered and ordered PageRank for the same
tolerance and tolerance function. We use a damping factor of `α = 0.85` and
limit the maximum number of iterations to `L = 500`. The error between the
approaches is calculated with *L1-norm*. The *sequential* *unordered* approach
is considered to be the *gold standard* (wrt to which error is measured). *Dead
ends* in the graph are handled by self loops to all the vertices (*loopall*
approach [(6)]).

From the results, we observe that **monolithic approaches are faster** for
OpenMP-based unordered/unordered approaches, and **levelwise approaches are**
**faster** for barrier-free approaches with full/partial error measurement. While
we do see that **levelwise barrier-free approach with partial error**
**measurement** is the **fastest** both in terms of *time* and in the number of
*iterations* the difference is generally small. However when performing PageRank
computation in a distributed setting, we expect levelwise based approaches to
provide a good advantage, thanks to its reduced communication requirement.

All outputs are saved in a [gist] and a small part of the output is listed here.
Some [charts] are also included below, generated from [sheets]. The input data
used for this experiment is available from the [SuiteSparse Matrix Collection].
This experiment was done with guidance from [Prof. Kishore Kothapalli],
[Prof. Dip Sankar Banerjee], and [Prof. Sathya Peri].

<br>

```bash
$ g++ -std=c++17 -O3 -fopenmp main.cxx
$ ./a.out ~/data/web-Stanford.mtx
$ ./a.out ~/data/web-BerkStan.mtx
$ ...

# Loading graph /home/subhajit/data/web-Stanford.mtx ...
# order: 281903 size: 2312497 [directed] {}
# order: 281903 size: 2594400 [directed] {} (selfLoopAllVertices)
# order: 281903 size: 2594400 [directed] {} (transposeWithDegree)
# OMP_NUM_THREADS=12
# - components: 29914
# - blockgraph-levels: 29914
# [00002.184 ms; 001 iters.] [0.0000e+00 err.] pagerankOmpUnordered              {tol_norm: Li, tolerance: 1e-01}
# [00001.082 ms; 001 iters.] [0.0000e+00 err.] pagerankOmpUnorderedMonolithic    {tol_norm: Li, tolerance: 1e-01}
# [00002.499 ms; 001 iters.] [7.5867e-01 err.] pagerankOmpUnorderedLevelwise     {tol_norm: Li, tolerance: 1e-01}
# [00002.035 ms; 001 iters.] [1.6955e-01 err.] pagerankOmpOrdered                {tol_norm: Li, tolerance: 1e-01}
# [00001.298 ms; 001 iters.] [2.1292e-01 err.] pagerankOmpOrderedMonolithic      {tol_norm: Li, tolerance: 1e-01}
# [00002.779 ms; 001 iters.] [2.1281e-01 err.] pagerankOmpOrderedLevelwise       {tol_norm: Li, tolerance: 1e-01}
# [00002.882 ms; 001 iters.] [1.7175e-01 err.] pagerankBarrierfreeFull           {tol_norm: Li, tolerance: 1e-01}
# [00002.987 ms; 001 iters.] [2.1087e-01 err.] pagerankBarrierfreeFullMonolithic {tol_norm: Li, tolerance: 1e-01}
# [00003.609 ms; 001 iters.] [1.9101e-01 err.] pagerankBarrierfreeFullLevelwise  {tol_norm: Li, tolerance: 1e-01}
# [00002.792 ms; 001 iters.] [1.7122e-01 err.] pagerankBarrierfreePart           {tol_norm: Li, tolerance: 1e-01}
# [00001.759 ms; 001 iters.] [2.1087e-01 err.] pagerankBarrierfreePartMonolithic {tol_norm: Li, tolerance: 1e-01}
# [00002.867 ms; 001 iters.] [1.9312e-01 err.] pagerankBarrierfreePartLevelwise  {tol_norm: Li, tolerance: 1e-01}
# [00003.469 ms; 003 iters.] [1.4471e-01 err.] pagerankOmpUnordered              {tol_norm: Li, tolerance: 1e-02}
# [00002.641 ms; 003 iters.] [1.4471e-01 err.] pagerankOmpUnorderedMonolithic    {tol_norm: Li, tolerance: 1e-02}
# [00003.392 ms; 002 iters.] [3.1873e-01 err.] pagerankOmpUnorderedLevelwise     {tol_norm: Li, tolerance: 1e-02}
# [00004.650 ms; 003 iters.] [2.1492e-01 err.] pagerankOmpOrdered                {tol_norm: Li, tolerance: 1e-02}
# [00002.076 ms; 002 iters.] [1.6446e-01 err.] pagerankOmpOrderedMonolithic      {tol_norm: Li, tolerance: 1e-02}
# [00004.128 ms; 002 iters.] [2.3940e-01 err.] pagerankOmpOrderedLevelwise       {tol_norm: Li, tolerance: 1e-02}
# [00005.145 ms; 002 iters.] [1.5048e-01 err.] pagerankBarrierfreeFull           {tol_norm: Li, tolerance: 1e-02}
# [00004.106 ms; 002 iters.] [1.8342e-01 err.] pagerankBarrierfreeFullMonolithic {tol_norm: Li, tolerance: 1e-02}
# [00004.411 ms; 002 iters.] [2.0031e-01 err.] pagerankBarrierfreeFullLevelwise  {tol_norm: Li, tolerance: 1e-02}
# [00003.623 ms; 001 iters.] [2.2269e-01 err.] pagerankBarrierfreePart           {tol_norm: Li, tolerance: 1e-02}
# [00002.139 ms; 001 iters.] [2.2705e-01 err.] pagerankBarrierfreePartMonolithic {tol_norm: Li, tolerance: 1e-02}
# [00003.264 ms; 001 iters.] [2.5809e-01 err.] pagerankBarrierfreePartLevelwise  {tol_norm: Li, tolerance: 1e-02}
# ...
# [00370.699 ms; 500 iters.] [0.0000e+00 err.] pagerankOmpUnordered              {tol_norm: Li, tolerance: 1e-14}
# [00266.984 ms; 500 iters.] [0.0000e+00 err.] pagerankOmpUnorderedMonolithic    {tol_norm: Li, tolerance: 1e-14}
# [00365.008 ms; 317 iters.] [1.2584e-07 err.] pagerankOmpUnorderedLevelwise     {tol_norm: Li, tolerance: 1e-14}
# [00123.909 ms; 098 iters.] [1.9161e-07 err.] pagerankOmpOrdered                {tol_norm: Li, tolerance: 1e-14}
# [00120.761 ms; 097 iters.] [2.1604e-07 err.] pagerankOmpOrderedMonolithic      {tol_norm: Li, tolerance: 1e-14}
# [00173.049 ms; 079 iters.] [2.2971e-07 err.] pagerankOmpOrderedLevelwise       {tol_norm: Li, tolerance: 1e-14}
# [00179.473 ms; 111 iters.] [2.0161e-07 err.] pagerankBarrierfreeFull           {tol_norm: Li, tolerance: 1e-14}
# [00104.218 ms; 113 iters.] [2.3566e-07 err.] pagerankBarrierfreeFullMonolithic {tol_norm: Li, tolerance: 1e-14}
# [00133.347 ms; 102 iters.] [2.4534e-07 err.] pagerankBarrierfreeFullLevelwise  {tol_norm: Li, tolerance: 1e-14}
# [00134.317 ms; 094 iters.] [1.9824e-07 err.] pagerankBarrierfreePart           {tol_norm: Li, tolerance: 1e-14}
# [00070.508 ms; 097 iters.] [2.4929e-07 err.] pagerankBarrierfreePartMonolithic {tol_norm: Li, tolerance: 1e-14}
# [00079.453 ms; 082 iters.] [4.8501e-04 err.] pagerankBarrierfreePartLevelwise  {tol_norm: Li, tolerance: 1e-14}
#
# Loading graph /home/subhajit/data/web-BerkStan.mtx ...
# order: 685230 size: 7600595 [directed] {}
# order: 685230 size: 8285825 [directed] {} (selfLoopAllVertices)
# order: 685230 size: 8285825 [directed] {} (transposeWithDegree)
# OMP_NUM_THREADS=12
# - components: 109406
# - blockgraph-levels: 109406
# [00003.491 ms; 001 iters.] [0.0000e+00 err.] pagerankOmpUnordered              {tol_norm: Li, tolerance: 1e-01}
# [00002.800 ms; 001 iters.] [0.0000e+00 err.] pagerankOmpUnorderedMonolithic    {tol_norm: Li, tolerance: 1e-01}
# [00007.115 ms; 001 iters.] [8.3842e-01 err.] pagerankOmpUnorderedLevelwise     {tol_norm: Li, tolerance: 1e-01}
# [00003.220 ms; 001 iters.] [2.4913e-01 err.] pagerankOmpOrdered                {tol_norm: Li, tolerance: 1e-01}
# [00003.234 ms; 001 iters.] [2.6340e-01 err.] pagerankOmpOrderedMonolithic      {tol_norm: Li, tolerance: 1e-01}
# [00007.980 ms; 001 iters.] [2.7101e-01 err.] pagerankOmpOrderedLevelwise       {tol_norm: Li, tolerance: 1e-01}
# [00005.328 ms; 001 iters.] [2.5949e-01 err.] pagerankBarrierfreeFull           {tol_norm: Li, tolerance: 1e-01}
# [00006.049 ms; 001 iters.] [2.5814e-01 err.] pagerankBarrierfreeFullMonolithic {tol_norm: Li, tolerance: 1e-01}
# [00007.284 ms; 001 iters.] [2.5112e-01 err.] pagerankBarrierfreeFullLevelwise  {tol_norm: Li, tolerance: 1e-01}
# [00004.339 ms; 001 iters.] [2.5959e-01 err.] pagerankBarrierfreePart           {tol_norm: Li, tolerance: 1e-01}
# [00004.803 ms; 001 iters.] [2.5795e-01 err.] pagerankBarrierfreePartMonolithic {tol_norm: Li, tolerance: 1e-01}
# [00006.115 ms; 001 iters.] [2.5123e-01 err.] pagerankBarrierfreePartLevelwise  {tol_norm: Li, tolerance: 1e-01}
# ...
```

[![](https://i.imgur.com/TaQ6tRH.png)][sheetp]
[![](https://i.imgur.com/93Bvchw.png)][sheetp]
[![](https://i.imgur.com/GYXa54D.png)][sheetp]

<br>
<br>


## References

- [An Efficient Practical Non-Blocking PageRank Algorithm for Large Scale Graphs; Hemalatha Eedi et al. (2021)](https://ieeexplore.ieee.org/document/9407114)
- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](https://www.youtube.com/watch?v=ke9g8hB0MEo)
- [The PageRank Citation Ranking: Bringing Order to the Web; Larry Page et al. (1998)](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.38.5427)
- [The University of Florida Sparse Matrix Collection; Timothy A. Davis et al. (2011)](https://doi.org/10.1145/2049662.2049663)
- [What's the difference between "static" and "dynamic" schedule in OpenMP?](https://stackoverflow.com/a/10852852/1413259)
- [OpenMP Dynamic vs Guided Scheduling](https://stackoverflow.com/a/43047074/1413259)

<br>
<br>


[![](https://i.imgur.com/xCXqbfU.jpg)](https://www.youtube.com/watch?v=IY1VxuN7A14)<br>
[![DOI](https://zenodo.org/badge/534899343.svg)](https://zenodo.org/badge/latestdoi/534899343)


[(1)]: https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.38.5427
[(2)]: https://github.com/puzzlef/pagerank-ordered-vs-unordered
[(3)]: https://ieeexplore.ieee.org/document/9407114
[(4)]: https://ieeexplore.ieee.org/document/9835216
[(5)]: https://gist.github.com/wolfram77/12e5a19ff081b2e3280d04331a9976ca
[(6)]: https://gist.github.com/wolfram77/94c38b9cfbf0c855e5f42fa24a8602fc
[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[Prof. Sathya Peri]: https://people.iith.ac.in/sathya_p/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
[OpenMP]: https://en.wikipedia.org/wiki/OpenMP
[PageRank algorithm]: https://en.wikipedia.org/wiki/PageRank
[link analysis]: https://en.wikipedia.org/wiki/Network_theory#Link_analysis
[gist]: https://gist.github.com/wolfram77/6e267f7b71a5359c91873cd799ee73e4
[charts]: https://imgur.com/a/DWKowTI
[sheets]: https://docs.google.com/spreadsheets/d/1GE6WFj3-UY9W99GmM-iCHiIaTOh8uJIx-awdScKKzHc/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vSIrE6AqQoYMgG4rlak6f2gS_fUcacOJrrjdJk7wKpGrfYqWWPB1jijpeCSyXEuUCdiUSOLgMed5GDA/pubhtml
