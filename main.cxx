#include <string>
#include <vector>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <iostream>
#include "src/main.hxx"

using namespace std;




// You can define datatype with -DTYPE=...
#ifndef TYPE
#define TYPE float
#endif
// You can define number of threads with -DMAX_THREADS=...
#ifndef MAX_THREADS
#define MAX_THREADS 12
#endif




template <class G, class H>
void runPagerank(const G& x, const H& xt, int repeat) {
  using T = TYPE;
  enum NormFunction { L0=0, L1=1, L2=2, Li=3 };
  vector<T> *init = nullptr;
  float damping   = 0.85;

  // Find Pagerank data.
  auto cs  = components(x, xt);
  auto b   = blockgraph(x, cs);
  auto bt  = transpose(b);
  auto gs  = levelwiseComponentsFrom(cs, b, bt);
  printf("- components: %zu\n", cs.size());
  printf("- blockgraph-levels: %d\n", gs.size());
  PagerankData<G> C {move(cs), move(b), move(bt)};

  // Use Li-norm for convergence check.
  for (float tolerance=1e-1; tolerance>=1e-15; tolerance/=10) {
    // Find pagerank using a single thread for reference (unordered, no dead ends).
    auto a0 = pagerankMonolithicSeq<false, false>(x, xt, init, {1, false, damping, Li, tolerance});
    do {
      // Find pagerank accelerated with OpenMP (unordered, no dead ends).
      auto a1 = pagerankMonolithicOmp<false, false>(x, xt, init, {repeat, false, damping, Li, tolerance}, &C);
      auto e1 = l1Norm(a1.ranks, a0.ranks);
      printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankOmpUnordered              {tol_norm: Li, tolerance: %.0e}\n", a1.time, a1.iterations, e1, tolerance);
    } while (0);
    do {
      // Find pagerank accelerated with OpenMP (unordered, no dead ends, monolithic).
      auto a1 = pagerankMonolithicOmp<false, false>(x, xt, init, {repeat, true, damping, Li, tolerance}, &C);
      auto e1 = l1Norm(a1.ranks, a0.ranks);
      printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankOmpUnorderedMonolithic    {tol_norm: Li, tolerance: %.0e}\n", a1.time, a1.iterations, e1, tolerance);
    } while (0);
    do {
      // Find pagerank accelerated with OpenMP (unordered, no dead ends, levelwise).
      auto a1 = pagerankLevelwiseOmp<false, false>(x, xt, init, {repeat, true, damping, Li, tolerance}, &C);
      auto e1 = l1Norm(a1.ranks, a0.ranks);
      printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankOmpUnorderedLevelwise     {tol_norm: Li, tolerance: %.0e}\n", a1.time, a1.iterations, e1, tolerance);
    } while (0);

    do {
      // Find pagerank accelerated with OpenMP (ordered, no dead ends).
      auto a1 = pagerankMonolithicOmp<true, false>(x, xt, init, {repeat, false, damping, Li, tolerance}, &C);
      auto e1 = l1Norm(a1.ranks, a0.ranks);
      printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankOmpOrdered                {tol_norm: Li, tolerance: %.0e}\n", a1.time, a1.iterations, e1, tolerance);
    } while (0);
    do {
      // Find pagerank accelerated with OpenMP (ordered, no dead ends, monolithic).
      auto a1 = pagerankMonolithicOmp<true, false>(x, xt, init, {repeat, true, damping, Li, tolerance}, &C);
      auto e1 = l1Norm(a1.ranks, a0.ranks);
      printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankOmpOrderedMonolithic      {tol_norm: Li, tolerance: %.0e}\n", a1.time, a1.iterations, e1, tolerance);
    } while (0);
    do {
      // Find pagerank accelerated with OpenMP (ordered, no dead ends, levelwise).
      auto a1 = pagerankLevelwiseOmp<true, false>(x, xt, init, {repeat, true, damping, Li, tolerance}, &C);
      auto e1 = l1Norm(a1.ranks, a0.ranks);
      printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankOmpOrderedLevelwise       {tol_norm: Li, tolerance: %.0e}\n", a1.time, a1.iterations, e1, tolerance);
    } while (0);

    do {
      // Find pagerank with barrier-free iterations accelerated with OpenMP (ordered, no dead ends, full error).
      auto a1 = pagerankMonolithicBarrierfreeOmp<true, false, true>(x, xt, init, {repeat, false, damping, Li, tolerance}, &C);
      auto e1 = l1Norm(a1.ranks, a0.ranks);
      printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreeFull           {tol_norm: Li, tolerance: %.0e}\n", a1.time, a1.iterations, e1, tolerance);
    } while (0);
    do {
      // Find pagerank with barrier-free iterations accelerated with OpenMP (ordered, no dead ends, full error, monolithic).
      auto a1 = pagerankMonolithicBarrierfreeOmp<true, false, true>(x, xt, init, {repeat, true, damping, Li, tolerance}, &C);
      auto e1 = l1Norm(a1.ranks, a0.ranks);
      printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreeFullMonolithic {tol_norm: Li, tolerance: %.0e}\n", a1.time, a1.iterations, e1, tolerance);
    } while (0);
    do {
      // Find pagerank with barrier-free iterations accelerated with OpenMP (ordered, no dead ends, full error, levelwise).
      auto a1 = pagerankLevelwiseBarrierfreeOmp<true, false, true>(x, xt, init, {repeat, true, damping, Li, tolerance}, &C);
      auto e1 = l1Norm(a1.ranks, a0.ranks);
      printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreeFullLevelwise  {tol_norm: Li, tolerance: %.0e}\n", a1.time, a1.iterations, e1, tolerance);
    } while (0);

    do {
      // Find pagerank with barrier-free iterations accelerated with OpenMP (ordered, no dead ends, partial error).
      auto a1 = pagerankMonolithicBarrierfreeOmp<true, false, false>(x, xt, init, {repeat, false, damping, Li, tolerance}, &C);
      auto e1 = l1Norm(a1.ranks, a0.ranks);
      printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreePart           {tol_norm: Li, tolerance: %.0e}\n", a1.time, a1.iterations, e1, tolerance);
    } while (0);
    do {
      // Find pagerank with barrier-free iterations accelerated with OpenMP (ordered, no dead ends, partial error, monolithic).
      auto a1 = pagerankMonolithicBarrierfreeOmp<true, false, false>(x, xt, init, {repeat, true, damping, Li, tolerance}, &C);
      auto e1 = l1Norm(a1.ranks, a0.ranks);
      printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreePartMonolithic {tol_norm: Li, tolerance: %.0e}\n", a1.time, a1.iterations, e1, tolerance);
    } while (0);
    do {
      // Find pagerank with barrier-free iterations accelerated with OpenMP (ordered, no dead ends, partial error, levelwise).
      auto a1 = pagerankLevelwiseBarrierfreeOmp<true, false, false>(x, xt, init, {repeat, true, damping, Li, tolerance}, &C);
      auto e1 = l1Norm(a1.ranks, a0.ranks);
      printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreePartLevelwise  {tol_norm: Li, tolerance: %.0e}\n", a1.time, a1.iterations, e1, tolerance);
    } while (0);
  }
}


int main(int argc, char **argv) {
  char *file = argv[1];
  int repeat = argc>2? stoi(argv[2]) : 5;
  printf("Loading graph %s ...\n", file);
  auto x  = readMtxOutDiGraph(file); println(x);
  auto fl = [](auto u) { return true; };
  selfLoopU(x, None(), fl); print(x); printf(" (selfLoopAllVertices)\n");
  auto xt = transposeWithDegree(x);  print(xt); printf(" (transposeWithDegree)\n");
  omp_set_num_threads(MAX_THREADS);
  printf("OMP_NUM_THREADS=%d\n", MAX_THREADS);
  runPagerank(x, xt, repeat);
  printf("\n");
  return 0;
}
