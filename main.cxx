#include <utility>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
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




void runPagerankBatch(const string& data, size_t batch, size_t skip, int repeat) {
  using K = int;
  using T = TYPE;
  enum NormFunction { L0=0, L1=1, L2=2, Li=3 };
  vector<T> ranksOld, ranksAdj;
  vector<T> *initStatic  = nullptr;
  vector<T> *initDynamic = &ranksAdj;
  auto fl = [](auto u) { return true; };
  float damping   = 0.85;
  float tolerance = 1e-8;

  OutDiGraph<K> xo;
  stringstream  stream(data);
  while (true) {
    // Lets skip some edges.
    if (!readSnapTemporalW(xo, stream, skip)) break;
    auto x  = selfLoop(xo, None(), fl);
    auto xt = transposeWithDegree(x);
    auto a0 = pagerankMonolithicSeq<false, true>(x, xt, initStatic, {1, false, damping, Li, tolerance});
    auto ksOld = vertexKeys(x);
    ranksOld   = a0.ranks;

    // Read batch to be processed.
    auto yo = duplicate(xo);
    if (!readSnapTemporalW(yo, stream, batch)) break;
    auto y  = selfLoop(yo, None(), fl);
    auto yt = transposeWithDegree(y);
    auto a1 = pagerankMonolithicSeq<false, true>(y, yt, initStatic, {1, false, damping, Li, tolerance});
    auto ks = vertexKeys(y);

    // Find Pagerank data.
    using G = decltype(y);
    auto cs = components(y, yt);
    auto b  = blockgraph(y, cs);
    auto bt = transpose(b);
    auto gs = levelwiseComponentsFrom(cs, b, bt);
    printf("- components: %zu\n", cs.size());
    printf("- blockgraph-levels: %d\n", gs.size());
    PagerankData<G> C {move(cs), move(b), move(bt)};

    do {
      do {
        // Find pagerank accelerated with OpenMP (static, unordered, no dead ends).
        auto a2 = pagerankMonolithicOmp<false, false>(y, yt, initStatic, {repeat, false, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankOmpUnorderedStatic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank accelerated with OpenMP (static, unordered, no dead ends, monolithic).
        auto a2 = pagerankMonolithicOmp<false, false>(y, yt, initStatic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankOmpUnorderedMonolithicStatic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank accelerated with OpenMP (static, unordered, no dead ends, levelwise).
        auto a2 = pagerankLevelwiseOmp<false, false>(y, yt, initStatic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankOmpUnorderedLevelwiseStatic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank accelerated with OpenMP (static, ordered, no dead ends).
        auto a2 = pagerankMonolithicOmp<true, false>(y, yt, initStatic, {repeat, false, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankOmpOrderedStatic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank accelerated with OpenMP (static, ordered, no dead ends, monolithic).
        auto a2 = pagerankMonolithicOmp<true, false>(y, yt, initStatic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankOmpOrderedMonolithicStatic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank accelerated with OpenMP (static, ordered, no dead ends, levelwise).
        auto a2 = pagerankLevelwiseOmp<true, false>(y, yt, initStatic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankOmpOrderedLevelwiseStatic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank with barrier-free iterations accelerated with OpenMP (static, ordered, no dead ends, full error).
        auto a2 = pagerankMonolithicBarrierfreeOmp<true, false, true>(y, yt, initStatic, {repeat, false, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreeFullStatic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank with barrier-free iterations accelerated with OpenMP (static, ordered, no dead ends, full error, monolithic).
        auto a2 = pagerankMonolithicBarrierfreeOmp<true, false, true>(y, yt, initStatic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreeFullMonolithicStatic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank with barrier-free iterations accelerated with OpenMP (static, ordered, no dead ends, full error, levelwise).
        auto a2 = pagerankLevelwiseBarrierfreeOmp<true, false, true>(y, yt, initStatic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreeFullLevelwiseStatic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank with barrier-free iterations accelerated with OpenMP (static, ordered, no dead ends, partial error).
        auto a2 = pagerankMonolithicBarrierfreeOmp<true, false, false>(y, yt, initStatic, {repeat, false, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreePartStatic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank with barrier-free iterations accelerated with OpenMP (static, ordered, no dead ends, partial error, monolithic).
        auto a2 = pagerankMonolithicBarrierfreeOmp<true, false, false>(y, yt, initStatic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreePartMonolithicStatic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank with barrier-free iterations accelerated with OpenMP (static, ordered, no dead ends, partial error, levelwise).
        auto a2 = pagerankLevelwiseBarrierfreeOmp<true, false, false>(y, yt, initStatic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreePartLevelwiseStatic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
    } while (0);

    // Adjust ranks for dynamic Pagerank.
    ranksAdj.resize(y.span());
    adjustRanks(ranksAdj, ranksOld, ksOld, ks, 0.0f, float(ksOld.size())/ks.size(), 1.0f/ks.size());

    do {
      do {
        // Find pagerank accelerated with OpenMP (naive dynamic, unordered, no dead ends).
        auto a2 = pagerankMonolithicOmp<false, false>(y, yt, initDynamic, {repeat, false, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankOmpUnorderedNaiveDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank accelerated with OpenMP (naive dynamic, unordered, no dead ends, monolithic).
        auto a2 = pagerankMonolithicOmp<false, false>(y, yt, initDynamic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankOmpUnorderedMonolithicNaiveDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank accelerated with OpenMP (naive dynamic, unordered, no dead ends, levelwise).
        auto a2 = pagerankLevelwiseOmp<false, false>(y, yt, initDynamic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankOmpUnorderedLevelwiseNaiveDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank accelerated with OpenMP (naive dynamic, ordered, no dead ends).
        auto a2 = pagerankMonolithicOmp<true, false>(y, yt, initDynamic, {repeat, false, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankOmpOrderedNaiveDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank accelerated with OpenMP (naive dynamic, ordered, no dead ends, monolithic).
        auto a2 = pagerankMonolithicOmp<true, false>(y, yt, initDynamic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankOmpOrderedMonolithicNaiveDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank accelerated with OpenMP (naive dynamic, ordered, no dead ends, levelwise).
        auto a2 = pagerankLevelwiseOmp<true, false>(y, yt, initDynamic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankOmpOrderedLevelwiseNaiveDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank with barrier-free iterations accelerated with OpenMP (naive dynamic, ordered, no dead ends, full error).
        auto a2 = pagerankMonolithicBarrierfreeOmp<true, false, true>(y, yt, initDynamic, {repeat, false, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreeFullNaiveDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank with barrier-free iterations accelerated with OpenMP (naive dynamic, ordered, no dead ends, full error, monolithic).
        auto a2 = pagerankMonolithicBarrierfreeOmp<true, false, true>(y, yt, initDynamic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreeFullMonolithicNaiveDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank with barrier-free iterations accelerated with OpenMP (naive dynamic, ordered, no dead ends, full error, levelwise).
        auto a2 = pagerankLevelwiseBarrierfreeOmp<true, false, true>(y, yt, initDynamic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreeFullLevelwiseNaiveDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank with barrier-free iterations accelerated with OpenMP (naive dynamic, ordered, no dead ends, partial error).
        auto a2 = pagerankMonolithicBarrierfreeOmp<true, false, false>(y, yt, initDynamic, {repeat, false, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreePartNaiveDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank with barrier-free iterations accelerated with OpenMP (naive dynamic, ordered, no dead ends, partial error, monolithic).
        auto a2 = pagerankMonolithicBarrierfreeOmp<true, false, false>(y, yt, initDynamic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreePartMonolithicNaiveDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank with barrier-free iterations accelerated with OpenMP (naive dynamic, ordered, no dead ends, partial error, levelwise).
        auto a2 = pagerankLevelwiseBarrierfreeOmp<true, false, false>(y, yt, initDynamic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreePartLevelwiseNaiveDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
    } while (0);

    do {
      do {
        // Find pagerank accelerated with OpenMP (dynamic, unordered, no dead ends).
        auto a2 = pagerankMonolithicOmpDynamic<false, false>(x, xt, y, yt, initDynamic, {repeat, false, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankOmpUnorderedDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank accelerated with OpenMP (dynamic, unordered, no dead ends, monolithic).
        auto a2 = pagerankMonolithicOmpDynamic<false, false>(x, xt, y, yt, initDynamic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankOmpUnorderedMonolithicDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank accelerated with OpenMP (dynamic, unordered, no dead ends, levelwise).
        auto a2 = pagerankLevelwiseOmpDynamic<false, false>(x, xt, y, yt, initDynamic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankOmpUnorderedLevelwiseDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank accelerated with OpenMP (dynamic, ordered, no dead ends).
        auto a2 = pagerankMonolithicOmpDynamic<true, false>(x, xt, y, yt, initDynamic, {repeat, false, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankOmpOrderedDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank accelerated with OpenMP (dynamic, ordered, no dead ends, monolithic).
        auto a2 = pagerankMonolithicOmpDynamic<true, false>(x, xt, y, yt, initDynamic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankOmpOrderedMonolithicDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank accelerated with OpenMP (dynamic, ordered, no dead ends, levelwise).
        auto a2 = pagerankLevelwiseOmpDynamic<true, false>(x, xt, y, yt, initDynamic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankOmpOrderedLevelwiseDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank with barrier-free iterations accelerated with OpenMP (dynamic, ordered, no dead ends, full error).
        auto a2 = pagerankMonolithicBarrierfreeOmpDynamic<true, false, true>(x, xt, y, yt, initDynamic, {repeat, false, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreeFullDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank with barrier-free iterations accelerated with OpenMP (dynamic, ordered, no dead ends, full error, monolithic).
        auto a2 = pagerankMonolithicBarrierfreeOmpDynamic<true, false, true>(x, xt, y, yt, initDynamic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreeFullMonolithicDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank with barrier-free iterations accelerated with OpenMP (dynamic, ordered, no dead ends, full error, levelwise).
        auto a2 = pagerankLevelwiseBarrierfreeOmpDynamic<true, false, true>(x, xt, y, yt, initDynamic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreeFullLevelwiseDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank with barrier-free iterations accelerated with OpenMP (dynamic, ordered, no dead ends, partial error).
        auto a2 = pagerankMonolithicBarrierfreeOmpDynamic<true, false, false>(x, xt, y, yt, initDynamic, {repeat, false, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreePartDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank with barrier-free iterations accelerated with OpenMP (dynamic, ordered, no dead ends, partial error, monolithic).
        auto a2 = pagerankMonolithicBarrierfreeOmpDynamic<true, false, false>(x, xt, y, yt, initDynamic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreePartMonolithicDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
      do {
        // Find pagerank with barrier-free iterations accelerated with OpenMP (dynamic, ordered, no dead ends, partial error, levelwise).
        auto a2 = pagerankMonolithicBarrierfreeOmpDynamic<true, false, false>(x, xt, y, yt, initDynamic, {repeat, true, damping, Li, tolerance}, &C);
        auto e2 = l1Norm(a2.ranks, a1.ranks);
        printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankBarrierfreePartLevelwiseDynamic\n", y.order(), y.size(), a2.time, a2.iterations, e2);
      } while (0);
    } while (0);

    // Now time to move on to next batch.
    xo = move(yo);
  }
}


void runPagerank(const string& data, int repeat) {
  size_t M = countLines(data), steps = 10;
  printf("Temporal edges: %zu\n\n", M);
  for (size_t batch=100; batch<=1000000; batch*=10) {
    size_t skip = max(int64_t(M/steps) - int64_t(batch), 0L);
    printf("# Batch size %.0e\n", double(batch));
    runPagerankBatch(data, batch, skip, repeat);
    printf("\n");
  }
}


int main(int argc, char **argv) {
  char *file = argv[1];
  int repeat = argc>2? stoi(argv[2]) : 5;
  printf("Using graph %s ...\n", file);
  string data = readFile(file);
  omp_set_num_threads(MAX_THREADS);
  printf("OMP_NUM_THREADS=%d\n", MAX_THREADS);
  runPagerank(data, repeat);
  printf("\n");
  return 0;
}
