Effect of using different values of tolerance with barrier-free iterations in
[OpenMP]-based ordered monolithic [PageRank algorithm] for [link analysis].

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
independently, *without* waiting (with a barrier) for other threads to complete an
iteration. This minimizes unnecessary waits and allows each thread to be on a
*different iteration number* (which may or may not be beneficial for convergence)
[(3)]. **Monolithic PageRank** is the standard PageRank computation where vertices
are grouped by *strongly connected components* (SCCs). This improves *locality* of
memory accesses and thus improves performance [(4)].

In this experiment, we perform two different approaches of barrier-free
iterations of *OpenMP-based ordered PageRank*; one in which each thread detects
convergence by measuring the difference between the previous and the current
ranks of all the vertices (**full**), and the other in which the difference is
measured between the previous and current ranks of only the subset of vertices
being processed by each thread (**part**). Both approahes are performed in either
the standard way (with vertices arranged in vertex-id order) or with the monolithic
approach (where vertices are grouped by SCCs). This is done while adjusting the
tolerance `τ` from `10^-1` to `10^-14` with three different tolerance functions:
`L1-norm`, `L2-norm`, and `L∞-norm`. We also compare it with OpenMP-based
unordered and ordered PageRank for the same tolerance and tolerance function. We
use a damping factor of `α = 0.85` and limit the maximum number of iterations to
`L = 500`. The error between the approaches is calculated with *L1-norm*. The
*sequential unordered* approach is considered to be the *gold standard* (wrt to
which error is measured). *Dead ends* in the graph are handled by self loops to
all the vertices (*loopall* approach [(5)]).

From the results, we observe that **monolithic approaches are faster** than
default approaches in all the cases. We also observe that **monolithic**
**barrier-free** **approach with partial error measurement** is the **fastest** in
terms of *time*, and in terms of *iterations* for tolerance below `10^-10`.
However, *barrier-free approach with full error measurement* appears to take the
most time.

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
# [00002.000 ms; 001 iters.] [0.0000e+00 err.] pagerankOmpUnordered         {tol_norm: Li, tolerance: 1e-01}
# [00001.313 ms; 001 iters.] [0.0000e+00 err.] pagerankOmpUnorderedSplit    {tol_norm: Li, tolerance: 1e-01}
# [00002.463 ms; 001 iters.] [1.6893e-01 err.] pagerankOmpOrdered           {tol_norm: Li, tolerance: 1e-01}
# [00001.440 ms; 001 iters.] [2.0933e-01 err.] pagerankOmpOrderedSplit      {tol_norm: Li, tolerance: 1e-01}
# [00003.552 ms; 001 iters.] [1.7475e-01 err.] pagerankBarrierfreeFull      {tol_norm: Li, tolerance: 1e-01}
# [00003.312 ms; 001 iters.] [2.1233e-01 err.] pagerankBarrierfreeFullSplit {tol_norm: Li, tolerance: 1e-01}
# [00002.862 ms; 001 iters.] [1.7442e-01 err.] pagerankBarrierfreePart      {tol_norm: Li, tolerance: 1e-01}
# [00002.156 ms; 001 iters.] [2.1389e-01 err.] pagerankBarrierfreePartSplit {tol_norm: Li, tolerance: 1e-01}
# [00004.065 ms; 003 iters.] [1.4471e-01 err.] pagerankOmpUnordered         {tol_norm: Li, tolerance: 1e-02}
# [00003.022 ms; 003 iters.] [1.4471e-01 err.] pagerankOmpUnorderedSplit    {tol_norm: Li, tolerance: 1e-02}
# [00005.161 ms; 003 iters.] [2.1469e-01 err.] pagerankOmpOrdered           {tol_norm: Li, tolerance: 1e-02}
# [00002.706 ms; 003 iters.] [2.2768e-01 err.] pagerankOmpOrderedSplit      {tol_norm: Li, tolerance: 1e-02}
# [00005.578 ms; 002 iters.] [1.3942e-01 err.] pagerankBarrierfreeFull      {tol_norm: Li, tolerance: 1e-02}
# [00004.147 ms; 002 iters.] [1.8295e-01 err.] pagerankBarrierfreeFullSplit {tol_norm: Li, tolerance: 1e-02}
# [00003.834 ms; 001 iters.] [2.1781e-01 err.] pagerankBarrierfreePart      {tol_norm: Li, tolerance: 1e-02}
# [00002.556 ms; 001 iters.] [2.3123e-01 err.] pagerankBarrierfreePartSplit {tol_norm: Li, tolerance: 1e-02}
# ...
# [00357.997 ms; 500 iters.] [0.0000e+00 err.] pagerankOmpUnordered         {tol_norm: Li, tolerance: 1e-14}
# [00272.684 ms; 500 iters.] [0.0000e+00 err.] pagerankOmpUnorderedSplit    {tol_norm: Li, tolerance: 1e-14}
# [00122.743 ms; 098 iters.] [1.9258e-07 err.] pagerankOmpOrdered           {tol_norm: Li, tolerance: 1e-14}
# [00122.689 ms; 097 iters.] [2.1349e-07 err.] pagerankOmpOrderedSplit      {tol_norm: Li, tolerance: 1e-14}
# [00164.363 ms; 105 iters.] [1.9717e-07 err.] pagerankBarrierfreeFull      {tol_norm: Li, tolerance: 1e-14}
# [00107.156 ms; 116 iters.] [2.3638e-07 err.] pagerankBarrierfreeFullSplit {tol_norm: Li, tolerance: 1e-14}
# [00121.699 ms; 094 iters.] [1.9780e-07 err.] pagerankBarrierfreePart      {tol_norm: Li, tolerance: 1e-14}
# [00070.617 ms; 101 iters.] [2.4256e-07 err.] pagerankBarrierfreePartSplit {tol_norm: Li, tolerance: 1e-14}
#
# Loading graph /home/subhajit/data/web-BerkStan.mtx ...
# order: 685230 size: 7600595 [directed] {}
# order: 685230 size: 8285825 [directed] {} (selfLoopAllVertices)
# order: 685230 size: 8285825 [directed] {} (transposeWithDegree)
# OMP_NUM_THREADS=12
# - components: 109406
# [00003.520 ms; 001 iters.] [0.0000e+00 err.] pagerankOmpUnordered         {tol_norm: Li, tolerance: 1e-01}
# [00002.835 ms; 001 iters.] [0.0000e+00 err.] pagerankOmpUnorderedSplit    {tol_norm: Li, tolerance: 1e-01}
# [00003.230 ms; 001 iters.] [2.4860e-01 err.] pagerankOmpOrdered           {tol_norm: Li, tolerance: 1e-01}
# [00003.457 ms; 001 iters.] [2.6182e-01 err.] pagerankOmpOrderedSplit      {tol_norm: Li, tolerance: 1e-01}
# [00004.738 ms; 001 iters.] [2.5959e-01 err.] pagerankBarrierfreeFull      {tol_norm: Li, tolerance: 1e-01}
# [00006.693 ms; 001 iters.] [2.5795e-01 err.] pagerankBarrierfreeFullSplit {tol_norm: Li, tolerance: 1e-01}
# [00003.847 ms; 001 iters.] [2.5953e-01 err.] pagerankBarrierfreePart      {tol_norm: Li, tolerance: 1e-01}
# [00005.514 ms; 001 iters.] [2.5779e-01 err.] pagerankBarrierfreePartSplit {tol_norm: Li, tolerance: 1e-01}
# [00006.124 ms; 003 iters.] [1.6107e-01 err.] pagerankOmpUnordered         {tol_norm: Li, tolerance: 1e-02}
# [00006.353 ms; 003 iters.] [1.6107e-01 err.] pagerankOmpUnorderedSplit    {tol_norm: Li, tolerance: 1e-02}
# [00004.624 ms; 002 iters.] [1.9437e-01 err.] pagerankOmpOrdered           {tol_norm: Li, tolerance: 1e-02}
# [00005.628 ms; 002 iters.] [2.0232e-01 err.] pagerankOmpOrderedSplit      {tol_norm: Li, tolerance: 1e-02}
# [00005.007 ms; 001 iters.] [2.5756e-01 err.] pagerankBarrierfreeFull      {tol_norm: Li, tolerance: 1e-02}
# [00007.768 ms; 003 iters.] [2.3774e-01 err.] pagerankBarrierfreeFullSplit {tol_norm: Li, tolerance: 1e-02}
# [00003.826 ms; 001 iters.] [2.5757e-01 err.] pagerankBarrierfreePart      {tol_norm: Li, tolerance: 1e-02}
# [00005.720 ms; 001 iters.] [2.6246e-01 err.] pagerankBarrierfreePartSplit {tol_norm: Li, tolerance: 1e-02}
# ...
```

[![](https://i.imgur.com/iPfCdjW.png)][sheetp]
[![](https://i.imgur.com/Htx47t1.png)][sheetp]
[![](https://i.imgur.com/gOb7CpB.png)][sheetp]

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


[![](https://i.imgur.com/oYAg9Ej.jpg)](http://www.youtube.com/watch?v=1jPkcs__S3s)<br>
[![DOI](https://zenodo.org/badge/534518352.svg)](https://zenodo.org/badge/latestdoi/534518352)


[(1)]: https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.38.5427
[(2)]: https://github.com/puzzlef/pagerank-ordered-vs-unordered
[(3)]: https://ieeexplore.ieee.org/document/9407114
[(4)]: https://ieeexplore.ieee.org/document/9835216
[(5)]: https://gist.github.com/wolfram77/94c38b9cfbf0c855e5f42fa24a8602fc
[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[Prof. Sathya Peri]: https://people.iith.ac.in/sathya_p/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
[OpenMP]: https://en.wikipedia.org/wiki/OpenMP
[PageRank algorithm]: https://en.wikipedia.org/wiki/PageRank
[link analysis]: https://en.wikipedia.org/wiki/Network_theory#Link_analysis
[gist]: https://gist.github.com/wolfram77/e59c7de7891b9ec0e718e638c7a34467
[charts]: https://imgur.com/a/bUhQpuz
[sheets]: https://docs.google.com/spreadsheets/d/1PemaP5XCeiBUhSX5bpQ8Lk7rQkFCRqVnOY6mzwZ3HoI/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vS75Vl2ekl7QtppOsz9GTo42Q6DL4hyiCQvOrAa3YEEMn_X-bQecHZGtaaKmIxFn2ThjSPQyZ7Ywi0d/pubhtml
