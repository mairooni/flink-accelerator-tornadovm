# flink-accelerator-tornadovm

A TornadoVM-backed `AcceleratorProvider` for Apache Flink's Table API.

Flink discovers an accelerator through `ServiceLoader` and knows nothing about the
one it finds — not the framework, not the device API, not the vendor. This
repository is the other side of that interface. It holds everything Flink must
not: kernel generation, buffer layouts, staging widths, and every constant
calibrated against a particular card.

## Why it is not part of Flink

It was, and the build gave the coupling away: a clean checkout of Flink needed an
artifact that is not on Maven Central, and the planner emitted
`uk.ac.manchester.tornado.api.types.arrays.DoubleArray` into generated source. An
extension point cannot be proposed on those terms, because the vendor *is* the
interface.

So the split is the point, not an accident of layout. Flink ships the work
description; this ships the rate and the kernel.

## What it needs

- **Flink with the accelerator SPI.** It is not in any release yet — the
  interfaces this implements (`AcceleratorProvider`, `AccelNode`, the IR) exist
  only on the `gpu-offload` branch. Build that with `mvn install -DskipTests`
  first.
- **TornadoVM 6.1.1-jdk21-dev**, built locally, matching `<tornado.version>` in
  the pom. Its own build installs `tornado-api`, `tornado-annotation`,
  `tornado-cublas` and `tornado-cudf` into `~/.m2`.
- **JDK 21.** TornadoVM's off-heap arrays are built on `java.lang.foreign`, a
  preview API on 21 and final on 22; Flink 2.3 has no JDK 22+ profile, so
  21-with-preview is what a TaskManager running this actually runs on.

## Build

    mvn install

## Deploy

Put `flink-accelerator-tornadovm/target/flink-accelerator-tornadovm-*.jar` in a
distribution's `lib/`, and start the TaskManagers with TornadoVM's JVM arguments.
`flink-accelerator-tornadovm/scripts/gpu-cluster-setup.sh` does both.

**[DEPLOYMENT.md](DEPLOYMENT.md)** has the whole picture: how the four pieces fit
together, the build order and why it is strict, what the setup script changes, and
the couplings that fail at runtime rather than at build time.

A TaskManager without the jar, without a device, or without those JVM arguments
runs the code-generated operator instead and logs why. Nothing fails.

## Layout

| module | what it is |
|---|---|
| `flink-accelerator-tornadovm` | the provider: kernel generation, gathers, the staging operator |
| `flink-accelerator-tornadovm-examples` | runnable examples and benchmarks, one jar each |

`flink-accelerator-tornadovm/FINDINGS.md` is the measurement record: where the
time actually goes, and which of the original predictions it contradicted.

## What each benchmark is evidence of

The examples do not all claim the same thing, and the difference is the difference
between a feature and a research result. Two kinds live here.

### Transparent Table API acceleration

The user writes ordinary SQL. There is no GPU-specific application code, no hint,
no annotation, no API — the provider is discovered on the TaskManager and decides
for itself. This is the claim the integration exists to support.

| example | what it shows |
|---|---|
| **`HaversineBenchmark`** | **the reference application**, and the primary transparent example. Enough transcendental arithmetic per row to offset staging, over an ordinary table. Its natural form has no `WHERE`. |
| `HaversineBenchmark --near` | **a mechanism test, not an application.** The `WHERE` exists to exercise the cuDF compaction; the threshold was chosen to make the path engage, not because anyone asked for readings within 2000 km. Use it to check the mechanism works end to end; do not quote it as a workload result. `SensorAnomalyBenchmark` is the application for that question. |
| **`SensorAnomalyBenchmark`** | **screening telemetry for anomalies**, and the honest home of the compaction question. Selectivity is intrinsic here rather than contrived — screening means almost everything is normal — so "does packing survivors on the device pay" has a real answer. Reports exact and floating-point columns side by side so membership and arithmetic can be judged apart. |
| `BlackScholesBenchmark` | a second arithmetic stress case. **Its normal CDF is a `tanh` approximation**, chosen because it is branchless and therefore expressible as one projection. It is not a financially authoritative pricing implementation and must not be quoted as one; what it measures is arithmetic throughput. Both paths use the same approximation, so the CPU/GPU comparison is exact even where the approximation is not. |
| `GroupedAggregateBenchmark`, `JoinBenchmark`, `SortBenchmark`, `OverAggregateBenchmark` | operator-seam tests. They show that the seam works and what it costs. They are **not** headline performance claims: row marshalling and whole-partition buffering dominate them. |
| `MixedTypeVerification` | correctness across column types, not speed. |

### Accelerator-native resident experiments

`KMeansBenchmark`, `LogisticRegressionBenchmark`, `FeatureGramBenchmark` and
`ReferenceJoinBenchmark` drive an engine (`KMeansEngine`, `LogisticRegressionEngine`,
`GpuGramEngine`, `ReferenceJoinEngine`) through direct transformations over data
that stays resident on the device between iterations.

**They are not transparent Table API acceleration and must never be reported as
if they were.** They are valuable TornadoVM results about a resident execution
model, and their very large multi-iteration gains are real for that model — but
they are not apples-to-apples with a SQL job that re-reads and rebuilds its state
on every iteration, which is what the CPU side of such a comparison does.

A figure from one of these is only meaningful alongside all three of:

1. standard SQL / multi-job execution;
2. the equivalent **CPU-resident** algorithm;
3. the **GPU-resident** algorithm.

Quoting (3) against (1) measures the execution model, not the device.

If resident iterative work becomes a product direction it needs its own proposal —
an explicit accelerator-native operator or library, with distribution, reduction,
state, checkpoint, precision and termination semantics stated. It is not an
extension of the offload path and should not be presented as one.

### Reading a compaction result

The device compaction is worth exactly what the rows it discards would have cost
to copy back, so its value is a function of **selectivity** and nothing else. At
10% survival it moves a tenth of the bytes out; at 90% it moves nine tenths and
still pays for three extra device kernels a batch. A benchmark that does not state
its selectivity has not reported the result.

Two numbers make a compaction claim checkable, and both are printed: the bytes
actually moved (`OffloadMetrics`, in the TaskManager log) and the survivor count.
`TORNADOVM_ACCELERATOR_COMPACTION=false` runs the identical workload with the host
drain instead, which is the only way to attribute a difference to the compaction
rather than to the device.

### What every reported number must carry

`BenchmarkRun` collects it and the benchmarks print it: exact commits and build,
JVM flags, CUDA and driver versions, GPU and CPU model, clocks and power limit,
host load, dataset checksum and shape, parallelism and batch size, warm-up policy,
repetitions, and median / p95 / min / max / stddev with the cold run reported
separately. A time without that beside it is an anecdote.
