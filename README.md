Design of OpenMP-based *statically scheduled Barrier-free* **Dynamic** PageRank
algorithm for link analysis.

**Unordered PageRank** is the *standard* approach of PageRank computation (as
described in the original paper by Larry Page et al. [(1)][page]), where *two*
*different rank vectors* are maintained; one representing the *current* ranks of
vertices, and the other representing the *previous* ranks. On the other hand,
**ordered PageRank** uses *a single rank vector*, representing the current ranks
of vertices [(2)][pagerank]. This is similar to barrierless non-blocking
implementations of the PageRank algorithm by Hemalatha Eedi et al. [(3)][eedi].
As ranks are updated in the same vector (with each iteration), the order in
which vertices are processed *affects* the final result (hence the adjective
*ordered*). However, as PageRank is an iteratively converging algorithm, results
obtained with either approach are *mostly the same*. **Barrier-free PageRank**
is an *ordered* *PageRank* where each thread processes a subset of vertices in
the graph independently, *without* waiting (with a barrier) for other threads to
complete an iteration. This minimizes unnecessary waits and allows each thread
to be on a *different iteration number* (which may or may not be beneficial for
convergence) [(3)][eedi]. **Monolithic PageRank** is the standard PageRank
computation where vertices are grouped by *strongly connected components*
(SCCs). This improves *locality* of memory accesses and thus improves
performance [(4)][sahu]. **Levelwise PageRank** is a decomposed form of PageRank
computation, where each *SCC* is processed by *topological order* in the
*block-graph* (all components in a *level* are processed together *until*
convergence, after which we proceed to the next level). This decomposition
allows for distributed computation *without per-iteration communication*.
However, it does not work on a graph which includes *dead ends* (vertices with
no outgoing edges, also called dangling nodes) [(5)][sahu].

Dynamic graphs, which change with time, have many applications. Computing ranks
of vertices from scratch on every update (*static PageRank*) may not be good
enough for an *interactive system*. In such cases, we only want to process ranks
of vertices which are likely to have changed. To handle any new vertices
added/removed, we first *adjust* the *previous ranks* (before the graph
update/batch) with a *scaled 1/N-fill* approach [(5)][pagerank-dynamic]. Then, with **naive**
**dynamic approach** we simply run the PageRank algorithm with the *initial ranks*
set to the adjusted ranks. Alternatively, with the (fully) **dynamic approach**
we first obtain a *subset of vertices* in the graph which are likely to be
affected by the update (using BFS/DFS from changed vertices), and then perform
PageRank computation on *only* this *subset of vertices*.

<br>


### Comparing with Monolithic approach

In this experiment ([approach-monolithic]), we compare the performance of
**static**, **naive dynamic**, and (fully) **barrier-free iterations in
dynamic** **OpenMP-based ordered PageRank** (along with similar
*unordered/ordered* *OpenMP-based approaches*). We take *temporal graphs* as
input, and add edges to our in-memory graph in batches of size `10^2 to 10^6`.
However we do *not* perform this on every point on the temporal graph, but
*only* on *5 time* *samples* of the graph (5 samples are good enough to obtain
an average). At each time sample we load `B` edges (where *B* is the batch
size), and perform *static*, *naive dynamic*, and *dynamic* PageRank. At each
time sample, each approach is performed *5 times* to obtain an average time for
that sample. We perform two different approaches of barrier-free iterations of
*OpenMP-based* *ordered PageRank*; one in which each thread detects convergence
by measuring the difference between the previous and the current ranks of all
the vertices (**full**), and the other in which the difference is measured
between the previous and current ranks of only the subset of vertices being
processed by each thread (**part**). A *schedule* of `dynamic, 2048` is used for
*OpenMP-based PageRank* as obtained in [(6)][pagerank-openmp]. We use the
follwing PageRank parameters: damping factor `α = 0.85`, tolerance `τ = 10^-6`,
and limit the maximum number of iterations to `L = 500.` The error between the
current and the previous iteration is obtained with *L∞-norm*, and is used to
detect convergence. *Dead ends* in the graph are handled by adding self-loops to
all vertices in the graph (*loopall* approach [(7)][teleport]). Error in ranks
obtained for each approach is measured relative to the *sequential static
approach* using *L1-norm*.

From the results we observe the following. Monolithic approaches are faster than
default approaches. With the default approach, dynamic and naive dynamic
approaches have similar performance. However, with the monolithic approach,
dynamic approach is definitely faster than naive dynamic. **Monolithic**
**OpenMP-based unordered dynamic PageRank** is the **fastest** among all the other
approaches in terms of **time**. *Monolithic barrier-free dynamic PageRank with*
*partial error measurement* is the *fastest* among all the other approaches in
terms of *iterations*.

[approach-monolithic]: https://github.com/puzzlef/pagerank-barrierfrees-openmp-dynamic/tree/approach-monolithic

<br>


### Comparing with Levelwise approach

In this experiment ([approach-levelwise]), we compare the performance of
**static**, **naive dynamic**, and (fully) **barrier-free iterations in**
**dynamic** **OpenMP-based ordered PageRank** (along with similar
*unordered/ordered* *OpenMP-based approaches*). We take *temporal graphs* as
input, and add edges to our in-memory graph in batches of size `10^2 to 10^6`.
However we do *not* perform this on every point on the temporal graph, but
*only* on *5 time* *samples* of the graph (5 samples are good enough to obtain
an average). At each time sample we load `B` edges (where *B* is the batch
size), and perform *static*, *naive dynamic*, and *dynamic* PageRank. At each
time sample, each approach is performed *5 times* to obtain an average time for
that sample. We perform two different approaches of barrier-free iterations of
*OpenMP-based* *ordered PageRank*; one in which each thread detects convergence
by measuring the difference between the previous and the current ranks of all
the vertices (**full**), and the other in which the difference is measured
between the previous and current ranks of only the subset of vertices being
processed by each thread (**part**). We also compare the *default*,
*monolithic*, and *levelwise* approaches. A *schedule* of `dynamic, 2048` is
used for *OpenMP-based PageRank* as obtained in [(1)][pagerank-openmp]. We use
the follwing PageRank parameters: damping factor `α = 0.85`, tolerance `τ = 10^-6`,
and limit the maximum number of iterations to `L = 500.` The error
between the current and the previous iteration is obtained with *L∞-norm*, and
is used to detect convergence. *Dead ends* in the graph are handled by adding
self-loops to all vertices in the graph (*loopall* approach [(2)][teleport]).
Error in ranks obtained for each approach is measured relative to the
*sequential static approach* using *L1-norm*.

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

[approach-levelwise]: https://github.com/puzzlef/pagerank-barrierfrees-openmp-dynamic/tree/approach-levelwise

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


[page]: https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.38.5427
[eedi]: https://ieeexplore.ieee.org/document/9407114
[sahu]: https://ieeexplore.ieee.org/document/9835216
[teleport]: https://gist.github.com/wolfram77/94c38b9cfbf0c855e5f42fa24a8602fc
[pagerank]: https://github.com/puzzlef/pagerank
[pagerank-dynamic]: https://github.com/puzzlef/pagerank-dynamic
[pagerank-openmp]: https://github.com/puzzlef/pagerank-openmp
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
