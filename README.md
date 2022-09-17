Performance of static vs dynamic barrier-free iterations in [OpenMP]-based
ordered levelwise [PageRank algorithm] for [link analysis].

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

Dynamic graphs, which change with time, have many applications. Computing ranks
of vertices from scratch on every update (*static PageRank*) may not be good
enough for an *interactive system*. In such cases, we only want to process ranks
of vertices which are likely to have changed. To handle any new vertices
added/removed, we first *adjust* the *previous ranks* (before the graph
update/batch) with a *scaled 1/N-fill* approach [(6)]. Then, with **naive**
**dynamic approach** we simply run the PageRank algorithm with the *initial ranks*
set to the adjusted ranks. Alternatively, with the (fully) **dynamic approach**
we first obtain a *subset of vertices* in the graph which are likely to be
affected by the update (using BFS/DFS from changed vertices), and then perform
PageRank computation on *only* this *subset of vertices*.

In this experiment, we compare the performance of **static**, **naive dynamic**,
and (fully) **barrier-free iterations in dynamic OpenMP-based ordered PageRank**
(along with similar *unordered/ordered OpenMP-based approaches*). We take
*temporal graphs* as input, and add edges to our in-memory graph in batches of
size `10^2 to 10^6`. However we do *not* perform this on every point on the
temporal graph, but *only* on *5 time samples* of the graph (5 samples are good
enough to obtain an average). At each time sample we load `B` edges (where *B*
is the batch size), and perform *static*, *naive dynamic*, and *dynamic*
PageRank. At each time sample, each approach is performed *5 times* to obtain an
average time for that sample. We perform two different approaches of
barrier-free iterations of *OpenMP-based ordered PageRank*; one in which each
thread detects convergence by measuring the difference between the previous and
the current ranks of all the vertices (**full**), and the other in which the
difference is measured between the previous and current ranks of only the subset
of vertices being processed by each thread (**part**). We also compare the
*default*, *monolithic*, and *levelwise* approaches. A *schedule* of `dynamic, 2048`
is used for *OpenMP-based PageRank* as obtained in [(7)]. We use the
follwing PageRank parameters: damping factor `α = 0.85`, tolerance `τ = 10^-6`,
and limit the maximum number of iterations to `L = 500.` The error between the
current and the previous iteration is obtained with *L∞-norm*, and is used to
detect convergence. *Dead ends* in the graph are handled by adding self-loops to
all vertices in the graph (*loopall* approach [(8)]). Error in ranks obtained
for each approach is measured relative to the *sequential static approach* using
*L1-norm*.

From the results, we make the following observations. Dynamic and naive dynamic
PageRank have similar performance, except for barrier-free approaches
(full/partial error measurement). Dynamic/naive dynamic levelwise approaches are
usually the fastest. Levelwise barrier-free approach with partial error
measurement is the fastest among naive dynamic approaches. **OpenMP-based**
**unordered monolithic/levelwise approaches are the fastest among dynamic**
**approaches (time)**. With respect to the number of iterations, levelwise
dynamic approches are usually the fastest, expect for barrier-free partial
approach where naive dynamic is faster. Among naive dynamic approaches,
barrier-free partial approach is the fastest (iterations). **Among dynamic**
**approaches, OpenMP-based ordered levelwise approach is the fastest**
**(iterations)**. Error is usually the highest for dynamic monolithic/levelwise
approaches, except for barrier-free partial approach. Among naive dynamic
approaches, barrier-free partial levelwise approach has the highest error.
**Among dynamic approaches, barrier-free partial monolithic/levelwise**
**approaches have the highest error**. When performing PageRank computation in a
distributed setting, we expect levelwise based approaches to provide a good
advantage, thanks to its reduced communication requirement.

All outputs are saved in a [gist] and a small part of the output is listed here.
Some [charts] are also included below, generated from [sheets]. The input data
used for this experiment is available from the [SuiteSparse Matrix Collection].
This experiment was done with guidance from [Prof. Kishore Kothapalli],
[Prof. Dip Sankar Banerjee], and [Prof. Sathya Peri].

<br>

```bash
$ g++ -std=c++17 -O3 -fopenmp main.cxx
$ ./a.out ~/data/email-Eu-core-temporal.txt
$ ./a.out ~/data/CollegeMsg.txt
$ ...

# Using graph /home/subhajit/data/email-Eu-core-temporal.txt ...
# OMP_NUM_THREADS=12
# Temporal edges: 332335
#
# # Batch size 1e+02
# - components: 168
# - blockgraph-levels: 168
# [751 order; 7703 size; 00000.641 ms; 063 iters.] [0.0000e+00 err.] pagerankOmpUnorderedStatic
# [751 order; 7703 size; 00000.618 ms; 063 iters.] [0.0000e+00 err.] pagerankOmpUnorderedMonolithicStatic
# [751 order; 7703 size; 00000.514 ms; 055 iters.] [5.6535e-06 err.] pagerankOmpUnorderedLevelwiseStatic
# [751 order; 7703 size; 00001.079 ms; 075 iters.] [5.6016e-07 err.] pagerankOmpOrderedStatic
# [751 order; 7703 size; 00001.050 ms; 075 iters.] [5.5120e-07 err.] pagerankOmpOrderedMonolithicStatic
# [751 order; 7703 size; 00000.518 ms; 042 iters.] [1.6026e-06 err.] pagerankOmpOrderedLevelwiseStatic
# [751 order; 7703 size; 00021.785 ms; 096 iters.] [6.3152e-07 err.] pagerankBarrierfreeFullStatic
# [751 order; 7703 size; 00006.849 ms; 069 iters.] [1.0251e-06 err.] pagerankBarrierfreeFullMonolithicStatic
# [751 order; 7703 size; 00005.151 ms; 055 iters.] [2.0347e-06 err.] pagerankBarrierfreeFullLevelwiseStatic
# [751 order; 7703 size; 00006.608 ms; 067 iters.] [2.0893e-06 err.] pagerankBarrierfreePartStatic
# [751 order; 7703 size; 00005.072 ms; 051 iters.] [6.2907e-06 err.] pagerankBarrierfreePartMonolithicStatic
# [751 order; 7703 size; 00004.076 ms; 041 iters.] [7.5648e-06 err.] pagerankBarrierfreePartLevelwiseStatic
# [751 order; 7703 size; 00000.651 ms; 057 iters.] [9.0126e-07 err.] pagerankOmpUnorderedNaiveDynamic
# [751 order; 7703 size; 00000.583 ms; 057 iters.] [9.0126e-07 err.] pagerankOmpUnorderedMonolithicNaiveDynamic
# [751 order; 7703 size; 00000.300 ms; 032 iters.] [4.2085e-06 err.] pagerankOmpUnorderedLevelwiseNaiveDynamic
# [751 order; 7703 size; 00000.851 ms; 057 iters.] [8.4747e-07 err.] pagerankOmpOrderedNaiveDynamic
# [751 order; 7703 size; 00000.823 ms; 057 iters.] [8.5015e-07 err.] pagerankOmpOrderedMonolithicNaiveDynamic
# [751 order; 7703 size; 00000.297 ms; 025 iters.] [1.6042e-06 err.] pagerankOmpOrderedLevelwiseNaiveDynamic
# [751 order; 7703 size; 00007.459 ms; 069 iters.] [8.2175e-07 err.] pagerankBarrierfreeFullNaiveDynamic
# [751 order; 7703 size; 00003.000 ms; 031 iters.] [1.3672e-06 err.] pagerankBarrierfreeFullMonolithicNaiveDynamic
# [751 order; 7703 size; 00002.310 ms; 032 iters.] [1.4559e-06 err.] pagerankBarrierfreeFullLevelwiseNaiveDynamic
# [751 order; 7703 size; 00003.671 ms; 037 iters.] [2.2024e-06 err.] pagerankBarrierfreePartNaiveDynamic
# [751 order; 7703 size; 00002.327 ms; 025 iters.] [6.5626e-06 err.] pagerankBarrierfreePartMonolithicNaiveDynamic
# [751 order; 7703 size; 00001.777 ms; 019 iters.] [8.6571e-06 err.] pagerankBarrierfreePartLevelwiseNaiveDynamic
# [751 order; 7703 size; 00000.583 ms; 057 iters.] [9.0126e-07 err.] pagerankOmpUnorderedDynamic
# [751 order; 7703 size; 00000.571 ms; 057 iters.] [9.0126e-07 err.] pagerankOmpUnorderedMonolithicDynamic
# [751 order; 7703 size; 00000.299 ms; 032 iters.] [4.2085e-06 err.] pagerankOmpUnorderedLevelwiseDynamic
# [751 order; 7703 size; 00000.820 ms; 057 iters.] [8.4745e-07 err.] pagerankOmpOrderedDynamic
# [751 order; 7703 size; 00000.809 ms; 057 iters.] [8.5012e-07 err.] pagerankOmpOrderedMonolithicDynamic
# [751 order; 7703 size; 00000.246 ms; 025 iters.] [1.6041e-06 err.] pagerankOmpOrderedLevelwiseDynamic
# [751 order; 7703 size; 00006.897 ms; 065 iters.] [8.2288e-07 err.] pagerankBarrierfreeFullDynamic
# [751 order; 7703 size; 00002.835 ms; 034 iters.] [1.5894e-06 err.] pagerankBarrierfreeFullMonolithicDynamic
# [751 order; 7703 size; 00002.758 ms; 033 iters.] [9.8988e-07 err.] pagerankBarrierfreeFullLevelwiseDynamic
# [751 order; 7703 size; 00003.656 ms; 037 iters.] [2.2422e-06 err.] pagerankBarrierfreePartDynamic
# [751 order; 7703 size; 00003.134 ms; 026 iters.] [7.4998e-06 err.] pagerankBarrierfreePartMonolithicDynamic
# [751 order; 7703 size; 00002.908 ms; 025 iters.] [7.5235e-06 err.] pagerankBarrierfreePartLevelwiseDynamic
# ...
# [986 order; 25915 size; 00004.010 ms; 017 iters.] [1.8110e-06 err.] pagerankBarrierfreePartDynamic
# [986 order; 25915 size; 00002.081 ms; 009 iters.] [1.3046e-05 err.] pagerankBarrierfreePartMonolithicDynamic
# [986 order; 25915 size; 00002.290 ms; 009 iters.] [1.2903e-05 err.] pagerankBarrierfreePartLevelwiseDynamic
#
# # Batch size 1e+03
# - components: 168
# - blockgraph-levels: 168
# [751 order; 7703 size; 00000.643 ms; 063 iters.] [0.0000e+00 err.] pagerankOmpUnorderedStatic
# [751 order; 7703 size; 00000.616 ms; 063 iters.] [0.0000e+00 err.] pagerankOmpUnorderedMonolithicStatic
# [751 order; 7703 size; 00000.504 ms; 055 iters.] [5.6535e-06 err.] pagerankOmpUnorderedLevelwiseStatic
# ...
# [986 order; 25915 size; 00018.795 ms; 064 iters.] [2.0370e-06 err.] pagerankBarrierfreePartDynamic
# [986 order; 25915 size; 00011.187 ms; 057 iters.] [1.2371e-05 err.] pagerankBarrierfreePartMonolithicDynamic
# [986 order; 25915 size; 00011.462 ms; 057 iters.] [1.1253e-05 err.] pagerankBarrierfreePartLevelwiseDynamic
#
#
# Using graph /home/subhajit/data/CollegeMsg.txt ...
# OMP_NUM_THREADS=12
# Temporal edges: 59836
#
# # Batch size 1e+02
# - components: 313
# - blockgraph-levels: 313
# [564 order; 2899 size; 00000.222 ms; 053 iters.] [0.0000e+00 err.] pagerankOmpUnorderedStatic
# [564 order; 2899 size; 00000.213 ms; 053 iters.] [0.0000e+00 err.] pagerankOmpUnorderedMonolithicStatic
# [564 order; 2899 size; 00000.235 ms; 056 iters.] [1.9589e-06 err.] pagerankOmpUnorderedLevelwiseStatic
# ...
```

[![](https://i.imgur.com/aiSwWhp.png)][sheetp]
[![](https://i.imgur.com/UfpQ2GW.png)][sheetp]
[![](https://i.imgur.com/EECtazu.png)][sheetp]
[![](https://i.imgur.com/kvnCjNZ.png)][sheetp]
[![](https://i.imgur.com/TJGbqHb.png)][sheetp]
[![](https://i.imgur.com/phCFOC1.png)][sheetp]

[![](https://i.imgur.com/gF0Wq95.png)][sheetp]
[![](https://i.imgur.com/WT0v8ol.png)][sheetp]
[![](https://i.imgur.com/lfnmPSm.png)][sheetp]
[![](https://i.imgur.com/QHOQwx3.png)][sheetp]
[![](https://i.imgur.com/k0FQd6G.png)][sheetp]
[![](https://i.imgur.com/NbP2RJY.png)][sheetp]

[![](https://i.imgur.com/jT9SP2D.png)][sheetp]
[![](https://i.imgur.com/mGPrqFi.png)][sheetp]
[![](https://i.imgur.com/Dju02yQ.png)][sheetp]
[![](https://i.imgur.com/Ks9hCS7.png)][sheetp]
[![](https://i.imgur.com/3zuCCtw.png)][sheetp]
[![](https://i.imgur.com/GJSGy5k.png)][sheetp]

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


[![](https://i.imgur.com/szTY38M.jpg)](https://www.youtube.com/watch?v=NYbeosJvOXI)<br>
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
[![DOI](https://zenodo.org/badge/535638829.svg)](https://zenodo.org/badge/latestdoi/535638829)


[(1)]: https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.38.5427
[(2)]: https://github.com/puzzlef/pagerank-ordered-vs-unordered
[(3)]: https://ieeexplore.ieee.org/document/9407114
[(4)]: https://ieeexplore.ieee.org/document/9835216
[(5)]: https://gist.github.com/wolfram77/12e5a19ff081b2e3280d04331a9976ca
[(6)]: https://gist.github.com/wolfram77/eb7a3b2e44e3c2069e046389b45ead03
[(7)]: https://github.com/puzzlef/pagerank-openmp-adjust-schedule
[(8)]: https://gist.github.com/wolfram77/94c38b9cfbf0c855e5f42fa24a8602fc
[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[Prof. Sathya Peri]: https://people.iith.ac.in/sathya_p/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
[OpenMP]: https://en.wikipedia.org/wiki/OpenMP
[PageRank algorithm]: https://en.wikipedia.org/wiki/PageRank
[link analysis]: https://en.wikipedia.org/wiki/Network_theory#Link_analysis
[gist]: https://gist.github.com/wolfram77/a26aa0a428c7603810921c2f08c27413
[charts]: https://imgur.com/a/AHCZk8r
[sheets]: https://docs.google.com/spreadsheets/d/1p0nxbgi26Ixofn2Zjg-zQ6_jVcoomPFf4AlZRdc572U/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vQBKAhcauukhWoEVvzTtnv4MTBhLsnGiJrmaRfT5xQ_BSiFt3m28OmHDEqoAkSycUMiK0KZ99gE_HPj/pubhtml
