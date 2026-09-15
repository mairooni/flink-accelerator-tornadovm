<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

# P1 — first measurements

Target query: `SELECT id, val * 2.0 + 1.0 AS scaled FROM t WHERE val > 0.5`
8M rows, `(BIGINT id, DOUBLE val)`, 50% selectivity.

Hardware: RTX 4070 Laptop (8 GiB), i9-13900H, OpenCL 3.0 / NVIDIA 595.71.05.
Software: Flink 2.3.0, TornadoVM 5.2.1-jdk21-dev (OpenCL backend), OpenJDK 21.0.2.

## Environment smoke check — passed

TornadoVM runs on JDK 21 with the shipped 176-line `tornado-argfile` (JVMCI flags, JPMS module
path, `--add-exports` into `jdk.internal.vm.ci`, `--enable-preview`). This clears the one delta
flagged in the design doc §10 — the prior integration ran on JDK 1.8.0_302, where none of the
module machinery existed.

Two things this surfaced that are worth recording:

- TornadoVM's native array types (`DoubleArray`, `IntArray`) are built on `java.lang.foreign`,
  which is **preview on JDK 21**. Compiling against them needs `--enable-preview`, and so does
  running. Already in the argfile; now also in the parent POM.
- `tornado-api` and `tornado-annotation` are not published to Maven Central for this build, so they
  are installed into the local repo from the dist via `install:install-file`. See `scripts/`.

## Correctness — passed

Both input tiers produce results identical to the scalar CPU reference. 0 mismatches over 8M rows,
including the selection mask and the compaction order.

## Performance — the GPU path loses, as predicted, but not for the predicted reason

Batch = 262,144. Single run, so treat ±20% as noise (see caveats).

| Segment    | tier2-binary | share | tier4-generic | share |
|------------|-------------:|------:|--------------:|------:|
| gather     |   112.901 ms | 61.4% |     85.053 ms | 55.6% |
| copy-in    |     7.766 ms |  4.2% |      8.516 ms |  5.6% |
| **kernel** |   **0.668 ms** | **0.4%** | **0.675 ms** | **0.4%** |
| copy-out   |     8.856 ms |  4.8% |      9.017 ms |  5.9% |
| drain      |    53.756 ms | 29.2% |     49.740 ms | 32.5% |
| **total**  | **183.9 ms** |       |  **153.0 ms** |       |

CPU baseline (scalar, single-threaded, same compaction work): **96.3 ms**.

So the offloaded path is **1.6–1.9× slower** than doing it on the CPU. That was the expected P1
outcome. What was *not* expected is the shape of the breakdown.

### The design doc's discriminator says the wrong thing here

The plan predicted copy-in dominance (77.4% in the prior VLDB work) and named device residency as
the fix. Measured here:

- **Device transfer is 9% combined.** copy-in 4.2% + copy-out 4.8%.
- **Kernel is 0.4%.**
- **Host-side staging is 90%.** gather 61% + drain 29%.

**P2 (device residency) can therefore buy at most ~9% on this workload.** It cannot close a 1.9×
gap. The bottleneck moved from the interconnect to the host-side row↔column conversion.

This is not a contradiction of the prior work — it is what happens once the marshalling that
dominated in 2022 (endianness reversal, padding, object flattening) is free, and once the
interconnect is PCIe 4 rather than PCIe 3. What is left exposed is the cost that was always there
underneath: touching 8M rows one at a time in Java.

### Batch size does not change the conclusion

| batch     | gather | copy-in | kernel | copy-out | drain | dispatch overhead |
|-----------|-------:|--------:|-------:|---------:|------:|------------------:|
| 65,536    |  46.4% |    3.6% |   0.6% |     5.2% | 44.2% |            51.5 ms |
| 262,144   |  61.4% |    4.2% |   0.4% |     4.8% | 29.2% |            37.1 ms |
| 1,048,576 |  45.5% |    3.8% |   0.2% |     4.6% | 45.9% |             9.0 ms |
| 4,194,304 |  58.9% |    4.3% |   0.5% |     5.7% | 30.7% |             3.6 ms |

Device work stays at ~9-10% throughout. Larger batches do cut TornadoVM dispatch overhead
(`execute()` wall minus attributed device time) by an order of magnitude — 51 ms → 3.6 ms — which
is worth having, but it is not where the 1.9× lives.

### Anomaly: tier 4 gathers *faster* than tier 2

`GenericRowData` (85 ms) beats `BinaryRowData` (113 ms), which inverts the expected ordering.
Probable cause is pointer-chase depth per row rather than field access cost:

- `BinaryRowData` → `MemorySegment[]` → `MemorySegment` → `byte[]`, plus `assertIndexIsValid` and
  an offset computation.
- `GenericRowData` → `Object[]` → `Double`.

**This number is probably pessimistic for tier 2 and should not be trusted yet.** The generator
allocates one `byte[]` per row, which is the worst case for locality. Real `BinaryRowData` batches
point into *shared* memory segments from the network stack or the sorter, so consecutive rows are
contiguous. Fixing the generator to allocate into one backing segment is the first thing to do
before drawing any tier conclusion.

## Caveats on these numbers

1. **Single runs.** Gather varied 68–113 ms across runs at different batch sizes with identical
   work. No warm-up iterations beyond one, no JMH, no confidence intervals.
2. **Tier 2 locality is unrealistic** (above).
3. **Rows arrive from an `ArrayList`**, so the gather includes a traversal over 8M distinct heap
   objects. Flink pushes records one at a time through `processElement`, frequently reusing the
   same object. The real gather may be cheaper; it will not be more expensive.
4. **Tier 1 is not really measured.** `ColumnarDoubleGather` currently reads per-row because
   `ColumnarRowData` exposes no getter for its `VectorizedColumnBatch`. The bulk path needs the
   accessor patch to `flink-table-common`. This is the tier most likely to change the verdict.
5. **The CPU baseline is single-threaded.** Flink would run it at task parallelism, making the
   real gap wider, not narrower.

## What this implies for the plan

Taking the numbers at face value, the ordering in the design doc is wrong for this workload:

- **P2 (residency) is not the next thing to do.** It addresses 9% of the time.
- **The next thing is tier 1** — a bulk columnar path that skips per-row staging entirely, which is
  the only change that attacks the 90%. That needs the `ColumnarRowData` accessor patch and a
  Parquet-backed input.
- **A 2-flop projection may simply be the wrong workload.** Kernel time is 0.4%; there is nothing
  for the GPU to win back. Before concluding anything about the architecture, the same harness
  should be run with an arithmetically heavier expression to see whether the device work can ever
  become a meaningful share.

Both of those are cheap to test with what is already built.

## Reproducing

```bash
mvn -o -DskipTests package
mvn -o -q dependency:build-classpath -Dmdep.outputFile=$PWD/target/cp.txt \
    -Dmdep.includeScope=test -pl flink-gpu-runtime
./scripts/run-harness.sh org.apache.flink.table.gpu.Harness 8000000 262144
```

> **`Harness` no longer exists.** It drove `FilterProjectEngine`, the fixed two-kernel catalogue
> that runtime generation replaced, and both were removed once nothing on the query path used them.
> The numbers above stand as recorded; the command does not run. `GeneratedKernelSweep` is the
> equivalent for kernel intensity. The gather-tier comparison has no replacement yet — when the
> bulk path is wired into `GpuCalcOperator` it should be measured against the operator itself
> rather than a standalone harness, which is what the operator actually does.

---

# Follow-up experiments

Two experiments run after the first measurements, both on 4M rows / batch 262,144 unless noted.

## Experiment 2 — tier 1 bulk columnar path

`BulkColumnarDoubleGather` reaches the `VectorizedColumnBatch` behind a `ColumnarRowData` (via
`VarHandle`, standing in for the accessor patch), detects runs of consecutive rows sharing a batch,
and copies each run into the off-heap staging buffer with a single `MemorySegment.copy`. Input is
real `HeapDoubleVector`/`HeapLongVector` columns in batches of `VectorizedColumnBatch.DEFAULT_SIZE`
(2048), the shape a vectorized Parquet reader produces.

Result: **100% of rows take the bulk path**, and gather falls from ~90 ms (per-row) to ~32 ms at
8M rows — roughly 3×.

### The earlier tier-2 anomaly was a harness bug, now fixed

The first run had `GenericRowData` gathering *faster* than `BinaryRowData`, which inverted the
expected ordering. Cause: the generator used `BinaryRowWriter`, which **allocates a fresh `byte[]`
per row in its constructor** and re-points the row at it — so every row was a separate object
reached through a four-deep pointer chase. Real post-shuffle rows point into shared network buffers
or sorter pages.

The generator now writes the binary layout directly into one shared segment at the 24-byte stride
(`bitset | id slot | val slot`). With that fixed the ordering is as predicted:

| tier | gather, 8M rows |
|------|----------------:|
| 1 columnar (bulk) |  31.8 ms |
| 2 binary (shared segment) |  62.2 ms |
| 4 generic |  90.3 ms |

## Experiment 1 — arithmetic intensity sweep

`heavyScaleAndSelect` adds `intensity` rounds of `sqrt(acc² + 1)·0.5 + acc·0.25` per element.
Only multiply, add and `sqrt` — all exactly-rounded in IEEE 754 — so host and device stay
comparable. Measured max relative error was **≤ 3.2e-16** (1–2 ULP) at every intensity, confirming
the device arithmetic is not drifting.

Total attributed time, milliseconds (lower is better):

| intensity | CPU baseline | tier 1 | tier 2 | tier 4 | kernel share (t1) |
|----------:|-------------:|-------:|-------:|-------:|------------------:|
|         0 |         40.0 |   73.8 |  103.0 |  120.8 |              0.4% |
|         4 |         41.6 |   49.4 |   70.3 |   62.0 |              3.9% |
|        16 |         94.7 |   42.7 |   61.6 |   71.3 |             16.1% |
|        64 |        628.7 |   61.7 |   81.0 |   87.6 |             43.6% |

Speedup over the CPU baseline:

| intensity | tier 1 | tier 2 | tier 4 |
|----------:|-------:|-------:|-------:|
|         0 |  0.54× |  0.39× |  0.33× |
|         4 |  0.84× |  0.59× |  0.67× |
|        16 |  **2.22×** |  **1.54×** |  **1.33×** |
|        64 | **10.19×** |  **7.76×** |  **7.18×** |

**There is a crossover, and it sits between intensity 4 and 16 — around 8 rounds, roughly 30-40
flops per row.** Below it the GPU path loses on every tier; above it every tier wins, including
tier 4.

## Revised conclusions

1. **The original target query is simply not offloadable-worth-it.** `val * 2.0 + 1.0` is 2 flops;
   kernel time was 0.4% and there was nothing to win back. This was a property of the workload, not
   of the architecture.
2. **Expression complexity, not just column type, has to gate the offload decision.** The design
   doc's `supportGpuOffload()` predicate only checks that every `RexNode` is arithmetic over
   fixed-width types. That is necessary but nowhere near sufficient: it would happily offload the
   2-flop query and lose 2×. The predicate needs a cost estimate over the `RexNode` tree with a
   floor around the measured crossover.
3. **The input tier sets the margin, not the verdict.** All three tiers cross over between the same
   two intensities. Tier 1 is worth roughly 1.4× over tier 2 at high intensity — real, but not the
   thing that decides whether offload works.
4. **Device residency (P2) still addresses only ~10%** of the time at any intensity measured, since
   copy-in + copy-out stayed at 9-12% throughout. It remains worth doing for multi-node subtrees,
   but it is not what unlocks the win.

## Additional caveats on the sweep

- **JIT warm-up confound.** Gather time for a given tier drifts downward across intensities within
  one JVM (tier 1: 16.9 → 12.8 → 12.7 → 11.8 ms) even though the gather work is identical. Later
  intensities benefit from earlier JIT compilation. Absolute gather numbers within a sweep should
  not be compared across intensities; the CPU-vs-GPU comparison at a fixed intensity is unaffected,
  since both arms run in the same JVM state.
- **Drain is large and noisy** — 25-63% of attributed time, varying more than 2× between tiers
  doing identical work. It is a candidate for the next round of attention, and possibly for the
  same bulk treatment as the gather.
- Still single runs, no JMH, no confidence intervals.

## Suggested next step

Before the planner fork: decide what the `supportGpuOffload()` cost floor should be, expressed in
something the planner can compute from a `RexNode` tree. The sweep gives an empirical anchor
(~30-40 flops per row on this hardware) but that number is device-specific and needs at least a
second data point.

---

# Cost model — derivation of the floor

The floor has to be expressed in the same weighted units the estimator produces, so the calibration
kernel is costed with the same `OperatorWeights` table. Its inner round —
`sqrt(acc*acc + 1.0)*0.5 + acc*0.25` — is 13 weighted units (SQRT 8, three multiplies, two adds),
over a base of 3 for the output projection and the filter comparison.

| intensity | raw ops | weighted ops/row | tier-1 speedup |
|----------:|--------:|-----------------:|---------------:|
| 0 | 3 | 3 | 0.54× |
| 4 | 27 | 55 | 0.84× |
| 16 | 99 | 211 | 2.22× |
| 64 | 387 | 835 | 10.19× |

Break-even brackets between **55 and 211** weighted ops/row. Log-interpolating between those two
points puts it at roughly **70**.

> **Correction to an earlier note in this document.** The first write-up said the crossover was
> "30–40 flops per row", and the design doc repeated it. That conflated *raw operation counts* with
> the estimator's *weighted* units, and also mis-read the bracket. The raw-op bracket is 27–99; the
> weighted bracket is 55–211. Only the weighted number is usable, because that is what
> `estimateRowCost()` returns.

`DEFAULT_MIN_ROW_COST` is set to **96**, about 1.4× above the interpolated break-even. This
knowingly declines expressions between 70 and 96 that would probably win. The asymmetry is
deliberate: a missed speedup is invisible, a silent 2× regression is a support incident. There is a
test (`floorGivesUpWinsBetweenBreakEvenAndFloor`) that pins this trade-off so it stays visible.

**The floor and the weight table are one artifact.** Changing a weight without re-deriving the floor
invalidates both, since the floor was computed by costing the calibration kernel with that table.

## Worked examples

| expression | weighted | verdict at floor 96 |
|---|---:|---|
| `val * 2.0 + 1.0` and `val > 0.5` | 3 | rejected — measured 0.54× |
| `POWER(EXP(v),2) + LN(v)*SIN(v)` | 86 | rejected (above break-even; conservative) |
| `POWER(EXP(v),2)*LN(v)*SIN(v)*COS(v)*ATAN(v)` | 134 | offloaded |
| anything over `DECIMAL` or `VARCHAR` | — | rejected, not expressible |

---

# End-to-end: a SQL query on the GPU

`SELECT id, val * 2.0 + 1.0 AS scaled FROM t WHERE val > N` planned by Flink, selected by the cost
gate, substituted for a TornadoVM kernel, executed on an RTX 4070. **Results identical to the same
query with offload disabled**, at 200k and 600k rows.

```
== GPU Offload ==
GPU  subtree 0: Calc(select=[id, ((val * 2.0) + 1.0) AS scaled], where=[(val > 100000.0)])
       3 weighted ops/row clears floor of 1

CPU rows=99,999   GPU rows=99,999
RESULTS IDENTICAL — GPU offload produced the same output as the CPU plan
```

The floor is lowered to 1 deliberately. The only shape the kernel catalogue matches is the two-flop
projection, which measures 0.54x; the expressions that clear the calibrated floor of 96 have no
kernel. **This proves the path, not a speedup.**

## Three defects the first run exposed

Each was invisible to every test that came before it, which is the argument for running the thing.

1. **`GpuCalcSpec` carried no output layout.** It described the kernel but not where its result
   goes, so an operator projecting `id, val*2.0+1.0` could not tell which output field was the
   pass-through and which was computed.
2. **`GpuCalcSpec` was not `Serializable`.** It travels inside the operator into the JobGraph;
   without it, `StreamGraph` serialization failed with *Could not serialize stream node Calc*.
3. **The comparison harness was wrong, not the offload.** The first run reported 100,024 CPU rows
   against 99,946 GPU rows. Cause: datagen's random generator is not seeded, so the two runs saw
   different data. Switching `val` to a sequence made the comparison meaningful. Worth recording
   because the symptom looked exactly like a correctness bug in the kernel.

## The gap that now defines the work

| | clears the cost floor | has a kernel |
|---|---|---|
| `val * 2.0 + 1.0` | no (cost 3) | **yes** |
| `EXP(val)*LN(val) + SIN(val)*COS(val) + POWER(val,3)` | **yes** (cost 109) | no |

Nothing is currently in both columns. The cost gate and the kernel catalogue are each correct and
they do not yet overlap. Closing that is the next milestone, and it is the fork in the road the
design document records: grow the catalogue by hand, or generate device code from the `RexNode`
tree and use `TaskGraph.prebuiltTask`.

## Reproducing

```bash
# forked planner must be installed locally
mvn -o -pl flink-table/flink-table-planner -DskipTests -Dfast install    # in the Flink checkout
mvn -o -DskipTests package                                               # here

D=$TORNADO_SDK
LIB=<flink-dist>/lib                 # excluding flink-table-planner-loader
java "@$D/tornado-argfile" -cp "<runtime classes>:<resources>:<forked planner jar>:$LIB/*" \
     org.apache.flink.table.gpu.EndToEnd 200000
```

---

# The kernel catalogue does not scale, and what to do about it

Probing the planner with thirteen query shapes, two offload:

| | shape | verdict |
|---|---|---|
| GPU | `SELECT id, val*2.0+1.0 WHERE val > 0.5` | 3 ops/row, clears floor |
| GPU | `SELECT id, val*2.0+1.0` | 2 ops/row, clears floor |
| cpu | filter on a different column than the projection | no kernel matches |
| cpu | `WHERE val < 0.5` | no kernel matches |
| cpu | two computed columns | no kernel matches |
| cpu | `val*2.0-1.0` | no kernel matches |
| cpu | `val/2.0` | no kernel matches |
| cpu | `EXP(val)*LN(val)` | no kernel matches |
| cpu | `n*2` on INT | provider declined; the kernel writes DOUBLE |
| cpu | DECIMAL / STRING | not expressible on device |
| cpu | `SUM(val)` | only Calc opts in |

The cost gate and the catalogue barely intersect: everything worth offloading has no kernel, and
the one kernel that exists is below the calibrated floor. Adding shapes by hand does not fix this —
the space of expressions is unbounded.

## Runtime kernel generation works (spike)

`spike/RuntimeKernelSpike` generates a kernel for an arbitrary expression, compiles it at runtime,
and runs it on the device. `EXP(v)*LN(v) + SIN(v)*COS(v)`, which has no catalogue entry:

```
javac available at runtime: true
elements=1024  compared=1024  non_finite=0  mismatches=0  max_rel_err=5.26e-15
ARBITRARY EXPRESSION RAN ON THE DEVICE, generated and compiled at runtime
```

The obstacle was always that TornadoVM resolves a kernel from its method reference's
`SerializedLambda`, which needs a `writeReplace()` only `javac` emits — so Janino-generated classes
are unusable, and that is what forced a fixed catalogue.

**The compiler does not have to be Janino.** The spike generates one compilation unit holding both
the kernel *and* the code that builds the `TaskGraph` referencing it, then compiles it with the
real `javac` through `javax.tools`. The method reference is then an ordinary javac lambda with a
proper `writeReplace`, and TornadoVM accepts it.

### What this costs

- **A JDK, not a JRE**, at run time: `ToolProvider.getSystemJavaCompiler()` returns null on a JRE.
  Not a new constraint here — TornadoVM already requires a JVMCI-enabled JDK.
- **javac is slower than Janino.** Paid once per distinct expression per TaskManager, alongside
  TornadoVM's own kernel compilation, not per batch.
- **Generated names must avoid OpenCL keywords.** The first spike run failed with *"Java method
  name corresponds to an OpenCL Token. Change the Java method's name: kernel"*. Any generator needs
  a reserved-word list.
- Generated classes need `--enable-preview -source 21`, since the kernel touches TornadoVM's
  off-heap array types.

### Implemented upstream: naming a kernel by Method

TornadoVM now accepts a `java.lang.reflect.Method` directly:

```java
TaskGraph task(String id, Method method, Object... args)
```

on branch `method-task-api` of the TornadoVM fork. The plumbing was already there — `createTask`
has an `extractCVs` flag, and with it false the lambda argument is unused; `scalaTask` already took
that path. Two call sites had to learn about it: `TornadoTaskGraph.addTask`, and
`ReduceCodeAnalysis`, which additionally indexed task parameters at `[i + 1]` to skip the lambda
slot that a method task does not have.

With it, **only the kernel is generated** — the `TaskGraph` is built normally from a `Method`
obtained by reflection. The spike now does exactly that, and still produces
`max_rel_err=5.26e-15` for `EXP(v)*LN(v) + SIN(v)*COS(v)`.

Two requirements the API documents, both consequences of how the method is reached:

- it must be **static**, since a `Method` carries no receiver;
- its declaring class's bytecode must be readable via `getResourceAsStream`, because that is how
  `ReflectionUniverse` obtains it. A class written to disk and loaded by a `URLClassLoader`
  qualifies; a purely in-memory class does not unless its loader also serves it as a resource.

That second point is what decides the compiler. javac-to-a-directory works as-is. Janino would work
too — its inability to emit `writeReplace` no longer matters — but only behind a class loader that
serves the generated bytes from `getResourceAsStream`.


---

# RexNode to Java: the generator

`plan/gpu/GpuKernelGenerator` turns a Calc's expressions into kernel source. It replaces
`GpuCalcMatcher` and the catalogue it matched against.

From the tree for `EXP(c0)*LN(c0) + SIN(c0)*COS(c0) + POWER(c0,3.0) WHERE c0 > 0.0` — a query the
catalogue could not express at all — it emits:

```java
public static void evaluate(DoubleArray c0_in, DoubleArray out0, IntArray mask) {
    for (@Parallel int i = 0; i < c0_in.getSize(); i++) {
        double c0 = c0_in.get(i);
        out0.set(i, (((TornadoMath.exp(c0) * TornadoMath.log(c0))
                    + (TornadoMath.sin(c0) * TornadoMath.cos(c0)))
                    + TornadoMath.pow(c0, 3.0)));
        if ((c0 > 0.0)) { mask.set(i, 1); } else { mask.set(i, 0); }
    }
}
```

Compiled and run through the new `Method` API on an RTX 4070: 4096 elements, 0 mismatches
(values and mask), `max_rel_err=1.42e-14`.

## Decisions worth knowing

**Only referenced columns are staged.** A projected-through column never reaches the device, so it
is absent from the kernel signature entirely. `SELECT id, f(val)` transfers `val` and not `id`.

**BIGINT is refused as an expression input.** Everything on the device is a double, which is exact
for `INTEGER`, `FLOAT` and `REAL` but not for `BIGINT`: values above 2^53 would come back different
from the CPU plan. A `BIGINT` column can still be *selected* — as a pass-through it keeps its exact
type, because it never crosses.

**OpenCL reserved words are checked, not discovered.** A method named `kernel` fails deep inside
TornadoVM's sketcher with *"Java method name corresponds to an OpenCL Token"*. The generator holds
a reserved-word list so that cannot reach a user.

**Filtering still writes a mask rather than compacting.** Compaction on device needs a prefix sum
and a scatter; the host walks the results anyway to build rows.

## Where the tests live, and why they are split

The generator's tests are in the planner and assert on the emitted source, not on its behaviour:
compiling it needs `tornado-api`, and the planner deliberately has no dependency on TornadoVM —
that separation is what lets the runtime be absent. Compiling and running generated kernels belongs
to the runtime module, which has TornadoVM but not Calcite. Neither module can do both, which is a
consequence of the layering rather than an oversight.

## Wired in: the thirteen shapes, re-measured

`CommonExecCalc` now generates a kernel instead of matching a catalogue, and `GpuCalcMatcher` is
gone. Re-running the same probe:

| | shape | before | after |
|---|---|---:|---:|
| projection + filter, same column | | GPU | GPU |
| projection only, no filter | | GPU | GPU |
| filter on a different column | | cpu | **GPU** |
| `WHERE val < 0.5` | | cpu | **GPU** |
| two computed columns | | cpu | **GPU** |
| `val*2.0-1.0` | | cpu | **GPU** |
| `val/2.0` | | cpu | **GPU** |
| `EXP(val)*LN(val)` | | cpu | **GPU** |
| `ORDER BY` above the Calc | | GPU | GPU |
| `n*2` on an INT column | | cpu | cpu |
| DECIMAL / STRING | | cpu | cpu |
| `SUM(val)` | | cpu | cpu |

**Two of thirteen became nine of thirteen.** The four that still do not offload are the ones that
should not: `DECIMAL` and `STRING` are not expressible on a device, `SUM` is an aggregate and only
`Calc` opts in, and INT-only arithmetic produces an INT column the kernel cannot write — the kernel
writes doubles, and returning a double where SQL says INTEGER would be wrong rather than slow.

Correctness, not just planning. A query the catalogue could never express —
`EXP(val/50000.0)*LN(val) + SIN(val)*COS(val) + POWER(val/1000.0,3.0), val/7.0 WHERE val > 2.0 AND
val < 120000.0` — run with offload on and off over 120k rows: **239,994 values compared, 0
non-finite, max relative error 0.00**, i.e. bit-identical to the CPU plan.

### One thing the first run got wrong

Compilation of the generated kernel failed with *"package uk.ac.manchester.tornado.api does not
exist"*. The generated source was correct; the compile classpath was not. Under TornadoVM's launch
configuration the API is on the JVM's **module** path, so `java.class.path` does not mention it.
`GeneratedKernelEngine` now builds the compile classpath from the code-source locations of the very
classes the kernel imports, which is where they actually came from whether that was the class path,
the module path, or a user-code loader inside a TaskManager.


---

# Intensity sweep through generated kernels

Re-running the calibration against real generated kernels rather than the synthetic knob.
4M rows, batch 262,144, RTX 4070. `gpu_ms` is everything the offload costs: gather, copy-in,
kernel, copy-out, drain.

| terms | weighted ops/row | cpu_ms | gpu_ms | speedup | kernel_ms |
|------:|-----------------:|-------:|-------:|--------:|----------:|
| 0 | 3 | 5 | 49 | 0.11× | 0.3 |
| 1 | 47 | 56 | 16 | **3.54×** | 1.6 |
| 2 | 91 | 132 | 19 | **7.01×** | 3.7 |
| 4 | 179 | 266 | 20 | **13.42×** | 5.7 |
| 8 | 355 | 546 | 25 | **21.62×** | 11.1 |
| 16 | 707 | — | — | failed | — |

Break-even, interpolating between 3 and 47, is around **17 weighted ops/row**.

## The floor of 96 is far too conservative — and worse, it is not a constant

The shipped default came from the synthetic sweep and put break-even near 70. This one puts it near
17. Both were measured on the same GPU, so the difference is not the device: the two sweeps use
different arithmetic. The synthetic kernel was `sqrt`-heavy; this one is `sin`/`cos`-heavy, and the
CPU is relatively much worse at transcendentals than at `sqrt` while the GPU is not.

**The cost floor therefore cannot be a single number calibrated once.** `OperatorWeights` assigns
ordinal weights that stand in for device work, but break-even depends on the *ratio* of CPU cost to
GPU cost, and that ratio differs per function. A floor of 96 rejects everything between 17 and 96,
which here is 3.5× to 7× of real speedup thrown away.

This is a genuine limitation of the model as designed, not a calibration error, and it should be
recorded as such. The options are a per-function CPU/GPU cost ratio rather than one ordinal weight,
or a floor set low enough to be safe across shapes and accepting occasional small regressions.

## Large expressions fail, and fail silently

At 16 terms (707 weighted) the kernel does not run:

```
[TornadoVM-OpenCL] asynchronous error: CL_OUT_OF_RESOURCES error executing
CL_COMMAND_NDRANGE_KERNEL on NVIDIA GeForce RTX 4070 Laptop GPU (Device 0).
```

Reducing the batch from 262,144 to 16,384 does not help, so it is per-thread register pressure from
the size of the expression rather than the number of threads.

**The error is asynchronous and nothing checks it.** The run reported `kernel_ms=0.0` and a speedup
of 81× — for work that never happened. An offload path that cannot tell a failed kernel from a fast
one is not safe to enable, so detecting this is a prerequisite to any performance claim, and the
generator likely needs an expression-size ceiling alongside the cost floor.

## A silent 1000× slowdown found on the way

The first run of this sweep reported the GPU getting *slower* as intensity rose: 49 s of kernel time
for 4M rows at 16 terms. The kernel was correct but sequential —

```c
i_3 = 0;
for (; i_3 < 262144;) { ... i_11 = i_3 + 1; ... }     // no get_global_id
```

`@Parallel` had been dropped. `ASMClassVisitor.getParallelAnnotations` located the kernel's class
file through `ClassLoader.getSystemClassLoader()`, which cannot see a class generated and loaded at
run time. Finding no annotations is not an error there — a class legitimately may have none — so
every device thread executed the whole loop, giving correct results about a thousand times slower
than it should.

Fixed in TornadoVM by looking through the declaring class's own loader first, which is what
`ReflectionUniverse` already does when it reads the same class's bytecode. The two had to agree: a
kernel whose bytecode could be read was still losing its annotations. The context class loader is
not a substitute, because annotation lookup runs on a sketcher worker thread that does not inherit
the submitting thread's.

**This invalidates the earlier spike result as a performance claim.** `RuntimeKernelSpike` reported
`max_rel_err=5.26e-15` and "ran on the device", which was true — but it was running sequentially,
and nothing in the output would have revealed it. Correctness was never in question; the speed
implied by "ran on the device" was.
