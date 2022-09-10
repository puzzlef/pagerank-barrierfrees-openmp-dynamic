Performance of static vs dynamic barrier-free iterations in [OpenMP]-based
ordered monolithic [PageRank algorithm] for [link analysis].

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

Dynamic graphs, which change with time, have many applications. Computing ranks
of vertices from scratch on every update (*static PageRank*) may not be good
enough for an *interactive system*. In such cases, we only want to process ranks
of vertices which are likely to have changed. To handle any new vertices
added/removed, we first *adjust* the *previous ranks* (before the graph
update/batch) with a *scaled 1/N-fill* approach [(5)]. Then, with **naive**
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
of vertices being processed by each thread (**part**). A *schedule* of
`dynamic, 2048` is used for *OpenMP-based PageRank* as obtained in [(6)]. We use
the follwing PageRank parameters: damping factor `α = 0.85`, tolerance `τ = 10^-6`,
and limit the maximum number of iterations to `L = 500.` The error between the
current and the previous iteration is obtained with *L∞-norm*, and is used to
detect convergence. *Dead ends* in the graph are handled by adding self-loops to
all vertices in the graph (*loopall* approach [(7)]). Error in ranks obtained
for each approach is measured relative to the *sequential static approach* using
*L1-norm*.

From the results we observe the following. Monolithic approaches are faster than
default approaches. With the default approach, dynamic and naive dynamic
approaches have similar performance. However, with the monolithic approach,
dynamic approach is definitely faster than naive dynamic. **Monolithic**
**OpenMP-based unordered dynamic PageRank** is the **fastest** among all the other
approaches in terms of **time**. *Monolithic barrier-free dynamic PageRank with*
*partial error measurement* is the *fastest* among all the other approaches in
terms of *iterations*.

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
# [751 order; 7703 size; 00000.620 ms; 063 iters.] [0.0000e+00 err.] pagerankOmpUnorderedStatic
# [751 order; 7703 size; 00000.611 ms; 063 iters.] [0.0000e+00 err.] pagerankOmpUnorderedSplitStatic
# [751 order; 7703 size; 00000.995 ms; 075 iters.] [5.6016e-07 err.] pagerankOmpOrderedStatic
# [751 order; 7703 size; 00000.969 ms; 075 iters.] [5.5120e-07 err.] pagerankOmpOrderedSplitStatic
# [751 order; 7703 size; 00013.024 ms; 079 iters.] [8.4596e-07 err.] pagerankBarrierfreeFullOmpStatic
# [751 order; 7703 size; 00004.907 ms; 070 iters.] [1.0242e-06 err.] pagerankBarrierfreeFullOmpSplitStatic
# [751 order; 7703 size; 00006.159 ms; 067 iters.] [2.0239e-06 err.] pagerankBarrierfreePartOmpStatic
# [751 order; 7703 size; 00003.288 ms; 054 iters.] [7.4488e-06 err.] pagerankBarrierfreePartOmpSplitStatic
# [751 order; 7703 size; 00000.586 ms; 057 iters.] [9.0126e-07 err.] pagerankOmpUnorderedNaiveDynamic
# [751 order; 7703 size; 00000.594 ms; 057 iters.] [9.0126e-07 err.] pagerankOmpUnorderedSplitNaiveDynamic
# [751 order; 7703 size; 00000.789 ms; 057 iters.] [8.4747e-07 err.] pagerankOmpOrderedNaiveDynamic
# [751 order; 7703 size; 00000.783 ms; 057 iters.] [8.5015e-07 err.] pagerankOmpOrderedSplitNaiveDynamic
# [751 order; 7703 size; 00008.093 ms; 065 iters.] [8.2594e-07 err.] pagerankBarrierfreeFullOmpNaiveDynamic
# [751 order; 7703 size; 00002.166 ms; 035 iters.] [9.9032e-07 err.] pagerankBarrierfreeFullOmpSplitNaiveDynamic
# [751 order; 7703 size; 00005.802 ms; 037 iters.] [2.2432e-06 err.] pagerankBarrierfreePartOmpNaiveDynamic
# [751 order; 7703 size; 00001.915 ms; 025 iters.] [5.9958e-06 err.] pagerankBarrierfreePartOmpSplitNaiveDynamic
# [751 order; 7703 size; 00000.571 ms; 057 iters.] [9.0126e-07 err.] pagerankOmpUnorderedDynamic
# [751 order; 7703 size; 00000.496 ms; 057 iters.] [5.5374e-04 err.] pagerankOmpUnorderedSplitDynamic
# [751 order; 7703 size; 00000.789 ms; 057 iters.] [8.4745e-07 err.] pagerankOmpOrderedDynamic
# [751 order; 7703 size; 00000.711 ms; 057 iters.] [5.5373e-04 err.] pagerankOmpOrderedSplitDynamic
# [751 order; 7703 size; 00006.619 ms; 070 iters.] [8.2242e-07 err.] pagerankBarrierfreeFullOmpDynamic
# [751 order; 7703 size; 00002.390 ms; 028 iters.] [5.5350e-04 err.] pagerankBarrierfreeFullOmpSplitDynamic
# [751 order; 7703 size; 00003.222 ms; 037 iters.] [2.2340e-06 err.] pagerankBarrierfreePartOmpDynamic
# [751 order; 7703 size; 00001.578 ms; 019 iters.] [5.5912e-04 err.] pagerankBarrierfreePartOmpSplitDynamic
# ...
# - components: 184
# [986 order; 25915 size; 00002.018 ms; 064 iters.] [0.0000e+00 err.] pagerankOmpUnorderedStatic
# [986 order; 25915 size; 00001.932 ms; 064 iters.] [0.0000e+00 err.] pagerankOmpUnorderedSplitStatic
# [986 order; 25915 size; 00002.729 ms; 072 iters.] [5.7090e-07 err.] pagerankOmpOrderedStatic
# [986 order; 25915 size; 00002.662 ms; 071 iters.] [6.2422e-07 err.] pagerankOmpOrderedSplitStatic
# [986 order; 25915 size; 00016.903 ms; 084 iters.] [1.0784e-06 err.] pagerankBarrierfreeFullOmpStatic
# [986 order; 25915 size; 00013.909 ms; 074 iters.] [2.1700e-06 err.] pagerankBarrierfreeFullOmpSplitStatic
# [986 order; 25915 size; 00012.626 ms; 064 iters.] [2.1229e-06 err.] pagerankBarrierfreePartOmpStatic
# [986 order; 25915 size; 00011.222 ms; 059 iters.] [1.3998e-05 err.] pagerankBarrierfreePartOmpSplitStatic
# [986 order; 25915 size; 00000.852 ms; 026 iters.] [1.3071e-06 err.] pagerankOmpUnorderedNaiveDynamic
# [986 order; 25915 size; 00000.822 ms; 026 iters.] [1.3071e-06 err.] pagerankOmpUnorderedSplitNaiveDynamic
# [986 order; 25915 size; 00001.029 ms; 026 iters.] [1.2276e-06 err.] pagerankOmpOrderedNaiveDynamic
# [986 order; 25915 size; 00000.986 ms; 026 iters.] [1.2205e-06 err.] pagerankOmpOrderedSplitNaiveDynamic
# [986 order; 25915 size; 00005.575 ms; 026 iters.] [1.2871e-06 err.] pagerankBarrierfreeFullOmpNaiveDynamic
# [986 order; 25915 size; 00002.617 ms; 018 iters.] [1.5635e-06 err.] pagerankBarrierfreeFullOmpSplitNaiveDynamic
# [986 order; 25915 size; 00003.129 ms; 017 iters.] [1.6378e-06 err.] pagerankBarrierfreePartOmpNaiveDynamic
# [986 order; 25915 size; 00001.884 ms; 009 iters.] [1.3737e-05 err.] pagerankBarrierfreePartOmpSplitNaiveDynamic
# [986 order; 25915 size; 00000.846 ms; 026 iters.] [1.3071e-06 err.] pagerankOmpUnorderedDynamic
# [986 order; 25915 size; 00000.363 ms; 012 iters.] [2.0509e-05 err.] pagerankOmpUnorderedSplitDynamic
# [986 order; 25915 size; 00001.025 ms; 026 iters.] [1.2276e-06 err.] pagerankOmpOrderedDynamic
# [986 order; 25915 size; 00000.334 ms; 009 iters.] [1.9841e-05 err.] pagerankOmpOrderedSplitDynamic
# [986 order; 25915 size; 00010.177 ms; 029 iters.] [1.2362e-06 err.] pagerankBarrierfreeFullOmpDynamic
# [986 order; 25915 size; 00003.340 ms; 012 iters.] [1.9782e-05 err.] pagerankBarrierfreeFullOmpSplitDynamic
# [986 order; 25915 size; 00005.796 ms; 018 iters.] [1.5470e-06 err.] pagerankBarrierfreePartOmpDynamic
# [986 order; 25915 size; 00002.524 ms; 008 iters.] [2.0574e-05 err.] pagerankBarrierfreePartOmpSplitDynamic
#
# # Batch size 1e+03
# - components: 168
# [751 order; 7703 size; 00000.613 ms; 063 iters.] [0.0000e+00 err.] pagerankOmpUnorderedStatic
# [751 order; 7703 size; 00000.591 ms; 063 iters.] [0.0000e+00 err.] pagerankOmpUnorderedSplitStatic
# [751 order; 7703 size; 00000.990 ms; 075 iters.] [5.6016e-07 err.] pagerankOmpOrderedStatic
# [751 order; 7703 size; 00000.960 ms; 075 iters.] [5.5120e-07 err.] pagerankOmpOrderedSplitStatic
# [751 order; 7703 size; 00009.586 ms; 086 iters.] [6.1220e-07 err.] pagerankBarrierfreeFullOmpStatic
# [751 order; 7703 size; 00006.841 ms; 069 iters.] [1.0254e-06 err.] pagerankBarrierfreeFullOmpSplitStatic
# [751 order; 7703 size; 00007.063 ms; 066 iters.] [2.1223e-06 err.] pagerankBarrierfreePartOmpStatic
# [751 order; 7703 size; 00004.869 ms; 052 iters.] [5.1990e-06 err.] pagerankBarrierfreePartOmpSplitStatic
# [751 order; 7703 size; 00000.610 ms; 060 iters.] [1.0349e-06 err.] pagerankOmpUnorderedNaiveDynamic
# [751 order; 7703 size; 00000.611 ms; 060 iters.] [1.0349e-06 err.] pagerankOmpUnorderedSplitNaiveDynamic
# [751 order; 7703 size; 00000.817 ms; 060 iters.] [9.6145e-07 err.] pagerankOmpOrderedNaiveDynamic
# [751 order; 7703 size; 00000.799 ms; 060 iters.] [9.7169e-07 err.] pagerankOmpOrderedSplitNaiveDynamic
# [751 order; 7703 size; 00010.341 ms; 068 iters.] [8.7693e-07 err.] pagerankBarrierfreeFullOmpNaiveDynamic
# [751 order; 7703 size; 00003.474 ms; 035 iters.] [1.0914e-06 err.] pagerankBarrierfreeFullOmpSplitNaiveDynamic
# [751 order; 7703 size; 00007.590 ms; 050 iters.] [1.9610e-06 err.] pagerankBarrierfreePartOmpNaiveDynamic
# [751 order; 7703 size; 00002.234 ms; 025 iters.] [4.2391e-06 err.] pagerankBarrierfreePartOmpSplitNaiveDynamic
# [751 order; 7703 size; 00000.591 ms; 060 iters.] [1.0360e-06 err.] pagerankOmpUnorderedDynamic
# [751 order; 7703 size; 00000.526 ms; 060 iters.] [3.0658e-03 err.] pagerankOmpUnorderedSplitDynamic
# [751 order; 7703 size; 00000.820 ms; 060 iters.] [9.6246e-07 err.] pagerankOmpOrderedDynamic
# [751 order; 7703 size; 00000.719 ms; 060 iters.] [3.0657e-03 err.] pagerankOmpOrderedSplitDynamic
# [751 order; 7703 size; 00007.922 ms; 073 iters.] [8.5548e-07 err.] pagerankBarrierfreeFullOmpDynamic
# [751 order; 7703 size; 00002.771 ms; 029 iters.] [3.0660e-03 err.] pagerankBarrierfreeFullOmpSplitDynamic
# [751 order; 7703 size; 00005.001 ms; 050 iters.] [1.9941e-06 err.] pagerankBarrierfreePartOmpDynamic
# [751 order; 7703 size; 00001.748 ms; 020 iters.] [3.0706e-03 err.] pagerankBarrierfreePartOmpSplitDynamic
#
#
# Using graph /home/subhajit/data/CollegeMsg.txt ...
# OMP_NUM_THREADS=12
# Temporal edges: 59836
#
# # Batch size 1e+02
# - components: 313
# [564 order; 2899 size; 00000.208 ms; 053 iters.] [0.0000e+00 err.] pagerankOmpUnorderedStatic
# [564 order; 2899 size; 00000.208 ms; 053 iters.] [0.0000e+00 err.] pagerankOmpUnorderedSplitStatic
# [564 order; 2899 size; 00000.356 ms; 067 iters.] [1.7122e-06 err.] pagerankOmpOrderedStatic
# [564 order; 2899 size; 00000.364 ms; 069 iters.] [1.6759e-06 err.] pagerankOmpOrderedSplitStatic
# [564 order; 2899 size; 00006.913 ms; 102 iters.] [1.2297e-06 err.] pagerankBarrierfreeFullOmpStatic
# [564 order; 2899 size; 00002.337 ms; 116 iters.] [1.9077e-06 err.] pagerankBarrierfreeFullOmpSplitStatic
# [564 order; 2899 size; 00002.929 ms; 064 iters.] [2.9538e-06 err.] pagerankBarrierfreePartOmpStatic
# [564 order; 2899 size; 00003.363 ms; 056 iters.] [4.9119e-06 err.] pagerankBarrierfreePartOmpSplitStatic
# [564 order; 2899 size; 00000.233 ms; 060 iters.] [1.8264e-06 err.] pagerankOmpUnorderedNaiveDynamic
# [564 order; 2899 size; 00000.238 ms; 060 iters.] [1.8264e-06 err.] pagerankOmpUnorderedSplitNaiveDynamic
# [564 order; 2899 size; 00000.327 ms; 061 iters.] [1.6754e-06 err.] pagerankOmpOrderedNaiveDynamic
# [564 order; 2899 size; 00000.320 ms; 061 iters.] [1.6729e-06 err.] pagerankOmpOrderedSplitNaiveDynamic
# [564 order; 2899 size; 00004.684 ms; 111 iters.] [1.4321e-06 err.] pagerankBarrierfreeFullOmpNaiveDynamic
# [564 order; 2899 size; 00002.344 ms; 120 iters.] [1.5515e-06 err.] pagerankBarrierfreeFullOmpSplitNaiveDynamic
# [564 order; 2899 size; 00001.998 ms; 052 iters.] [2.7515e-06 err.] pagerankBarrierfreePartOmpNaiveDynamic
# [564 order; 2899 size; 00000.723 ms; 042 iters.] [5.5871e-06 err.] pagerankBarrierfreePartOmpSplitNaiveDynamic
# [564 order; 2899 size; 00000.190 ms; 060 iters.] [7.2347e-04 err.] pagerankOmpUnorderedDynamic
# [564 order; 2899 size; 00000.144 ms; 060 iters.] [3.9270e-03 err.] pagerankOmpUnorderedSplitDynamic
# [564 order; 2899 size; 00000.262 ms; 061 iters.] [7.2346e-04 err.] pagerankOmpOrderedDynamic
# [564 order; 2899 size; 00000.208 ms; 061 iters.] [3.9270e-03 err.] pagerankOmpOrderedSplitDynamic
# [564 order; 2899 size; 00004.862 ms; 092 iters.] [7.2359e-04 err.] pagerankBarrierfreeFullOmpDynamic
# [564 order; 2899 size; 00002.653 ms; 094 iters.] [3.9272e-03 err.] pagerankBarrierfreeFullOmpSplitDynamic
# [564 order; 2899 size; 00002.826 ms; 051 iters.] [7.2385e-04 err.] pagerankBarrierfreePartOmpDynamic
# [564 order; 2899 size; 00000.721 ms; 031 iters.] [3.9303e-03 err.] pagerankBarrierfreePartOmpSplitDynamic
# ...
```

[![](https://i.imgur.com/37rJGjh.png)][sheetp]
[![](https://i.imgur.com/qauNm2R.png)][sheetp]
[![](https://i.imgur.com/vkiba2H.png)][sheetp]
[![](https://i.imgur.com/PENQ8um.png)][sheetp]
[![](https://i.imgur.com/ABxOjJz.png)][sheetp]

[![](https://i.imgur.com/54KAiyp.png)][sheetp]
[![](https://i.imgur.com/s1mYDll.png)][sheetp]
[![](https://i.imgur.com/gyqpxn3.png)][sheetp]
[![](https://i.imgur.com/ac23NF3.png)][sheetp]
[![](https://i.imgur.com/batwBYg.png)][sheetp]

[![](https://i.imgur.com/kk5z0EF.png)][sheetp]
[![](https://i.imgur.com/ppHz5zN.png)][sheetp]
[![](https://i.imgur.com/xC7Di3c.png)][sheetp]
[![](https://i.imgur.com/b0EUO5J.png)][sheetp]
[![](https://i.imgur.com/MSDDHIC.png)][sheetp]

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


[![](https://i.imgur.com/Fg73quJ.jpg)](https://www.youtube.com/watch?v=IY1VxuN7A14)<br>
[![DOI](https://zenodo.org/badge/534765115.svg)](https://zenodo.org/badge/latestdoi/534765115)


[(1)]: https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.38.5427
[(2)]: https://github.com/puzzlef/pagerank-ordered-vs-unordered
[(3)]: https://ieeexplore.ieee.org/document/9407114
[(4)]: https://ieeexplore.ieee.org/document/9835216
[(5)]: https://gist.github.com/wolfram77/eb7a3b2e44e3c2069e046389b45ead03
[(6)]: https://github.com/puzzlef/pagerank-openmp-adjust-schedule
[(7)]: https://gist.github.com/wolfram77/94c38b9cfbf0c855e5f42fa24a8602fc
[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[Prof. Sathya Peri]: https://people.iith.ac.in/sathya_p/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
[OpenMP]: https://en.wikipedia.org/wiki/OpenMP
[PageRank algorithm]: https://en.wikipedia.org/wiki/PageRank
[link analysis]: https://en.wikipedia.org/wiki/Network_theory#Link_analysis
[gist]: https://gist.github.com/wolfram77/ffef48d413ed5ba958f03eaa8340432b
[charts]: https://imgur.com/a/oP0M3k9
[sheets]: https://docs.google.com/spreadsheets/d/16F5TUYKK4nJGAXYxvP0uFJrIhuxUxUIPnytl9qQaY_s/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vTEf2ffkQQUY5i1MOOKnOHHQ_odxkNdwZYn_pWPD6I__yNsEtgfA_kOJZykItwVMXDTmzs_dZaJk_5M/pubhtml
