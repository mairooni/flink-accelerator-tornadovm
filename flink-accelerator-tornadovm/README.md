# Flink Table GPU runtime

Query-independent runtime for transparent GPU offload of Flink SQL, executed through TornadoVM.

The planner decides *what* to offload (`flink-table-planner`, package `plan.gpu`); this module
does it. The two never meet at compile time: the planner discovers
`GpuOperatorFactoryProvider` through `META-INF/services`, so it carries no reference to TornadoVM
and a cluster without this jar simply runs everything on the CPU.

## Contents

| | |
|---|---|
| `gather/` | staging strategies per input layout — columnar bulk copy, binary row transpose, generic accessors |
| `operator/` | `GeneratedKernelEngine` (javac, buffers, task graph) and `GpuCalcOperator` (the Flink operator) |
| `provider/` | the `ServiceLoader` entry point |
| `metrics/` | the gather / copy-in / kernel / copy-out breakdown |
| `GeneratedKernelSweep` | standalone benchmark, no Flink job — arithmetic-intensity sweep against the cost floor |

## Building

Part of the default build. TornadoVM is not published to Maven Central, so build a TornadoVM
distribution first -- its own build installs `tornado-api` and `tornado-annotation` into the local
repository under the coordinates this module asks for, and nothing else is needed:

```bash
cd $TORNADOVM_ROOT && bin/compile --jdk jdk21 --sdk --backend cuda
cd $FLINK_ROOT     && ./mvnw clean install -DskipTests -Djdk21 -Pjava21-target
```

No profile or extra argument is needed. The module was originally behind a `-Pgpu` profile; that
was removed because it kept the module out of the reactor and therefore out of IntelliJ's project
structure, where it appeared as loose class files with no configurable source root.

Note that `activeByDefault` would not have been a fix: Maven deactivates such profiles as soon as
any `-P` is given on the command line, and the documented build uses `-Pjava21-target`.

Use `clean install` when changing the TornadoVM version. The `writeReplace()` that lets TornadoVM
resolve a kernel from its method reference is emitted at compile time against a specific
`tornado-api`; Maven does not recompile unchanged sources for a version bump, and the stale classes
fail at run time with `Kernel entry ... has no writeReplace()`.

This module compiles at Java 21 with `--enable-preview`, overriding the repository-wide source
level 11: TornadoVM's off-heap array types are built on `java.lang.foreign`, a preview API on 21.

## Running

TornadoVM's JVM arguments must reach every JVM that touches the plan -- the client builds it, the
JobManager holds it, the TaskManager runs the kernel -- via `env.java.opts` in `conf/config.yaml`
or as VM options in an IDE. The SDK's `tornado-argfile.template` carries them; expand
`${TORNADOVM_HOME}` in it rather than copying the flags by hand, since it is generated from
`tornado --printJavaFlags` and stays correct across TornadoVM versions.

From an IDE, see `flink-examples/flink-examples-table/.../java/gpu/GpuOffloadExample.java`. It is a
correctness demonstration, not a performance one: its expression is two flops and it forces the
cost floor down to admit it.

### Benchmark on a standalone cluster

`GpuOffloadExample` runs in a MiniCluster, which hides everything deployment does -- per-subtask
kernel compilation, device contention, the client round trip. For numbers that reflect what a user
sees, `scripts/` drives a real cluster:

```bash
# build, once
./mvnw clean install -DskipTests -Djdk21 -Pjava21-target

export TORNADOVM_HOME=/path/to/tornadovm-sdk
DIST=$PWD/flink-dist/target/flink-2.3.0-bin/flink-2.3.0

flink-table/flink-table-gpu-runtime/scripts/gpu-cluster-setup.sh "$DIST"
flink-table/flink-table-gpu-runtime/scripts/run-haversine.sh    "$DIST" 2000000 1 10
#                                                                       rows  par runs
```

**There is no input file to fetch.** The first run generates it -- 2M rows into
`/tmp/flink-gpu-points-2000000-csv`, about three minutes, since `datagen` is the slow part at
roughly 100 us/row -- and later runs reuse it. Set `DATA` to put it elsewhere, `DEPOTS` to change
the number of reference points, `FORMAT=parquet` if Hadoop is on the cluster classpath. The 2M rows
are about 90 MB as CSV, which is why the file is generated rather than checked in.

`run-haversine.sh` starts the cluster, writes the input once if it is not already there, then runs
the query ten times with offload off and ten times with it on, through `bin/flink run` exactly as a
user would.

The query is **distance to the nearest of twenty reference points** -- what a logistics or retail
system asks of an address table. Two things about that are not incidental.

The input is read from a **file, not `datagen`**, which costs about 100 us per row and buries the
operator whatever the kernel does. Format is a parameter: `csv` by default because it works against
an unmodified distribution, `parquet` when Hadoop is on the cluster classpath, which hands the
operator `ColumnarRowData` and the columnar gather with it.

And the arithmetic has to be **worth offloading**. One distance is about 173 weighted ops against a
floor of 96 -- it clears the gate, but it is only a fifth of the job's wall time, so Amdahl caps the
whole query near 1.2x however fast the device is. Twenty of them is 3458 weighted ops with the input
still read once and still one output column.

Measured on an RTX 4070 Laptop, 2M rows at parallelism 1 with the csv source:

| depots | weighted ops/row | CPU | GPU | end to end |
|---|---|---|---|---|
| 1 | 173 | 1230 ms | 1070 ms | 1.15x |
| 20 | 3458 | 15400 ms | 1150 ms | 13.4x |

`--baseline` runs the same job with the arithmetic removed, which is how those shares were
established: reading and counting alone costs about 1000 ms, so at one depot the expression is 19%
of the job and at twenty it is 93%.

That table was taken in one sitting. Repeated later on a busier machine every absolute number
roughly doubled -- baseline 2180 ms, CPU 34900 ms, GPU 2440 ms -- but the ratio held at 14.3x,
because both sides move together. The governor here is `powersave`, so **only compare runs measured
back to back in the same cluster session**; numbers from different sittings are not comparable, and
neither are A/B variants that required a TaskManager restart between them.

The operator's own breakdown moves the same way. At one depot the drain is 72% of its time and the
kernel 20%; at twenty the kernel is 83% and the drain 15%.

That 72% is a misleading denominator, though, and worth not being fooled by: the operator is about
160 ms of a job whose other 2200 ms is reading the input, so the drain is under 2% of the job at
either setting. Replacing its `GenericRowData` with the `BoxedWrapperRowData` that Flink's own Calc
emits -- trading a boxed `Double` per value for a reused mutable wrapper -- was measured and made no
difference. The allocations are short-lived and die in the young generation; the wrapper trades them
for an array read, a checkcast and a null check. The change was reverted.

What is left to attack is the source, which is 89% of the offloaded job.

Results agree between the two paths to 2 ULP. They are not bit-identical and should not be expected
to be: the device reassociates floating-point arithmetic differently.

`FINDINGS.md` has the measurements behind the cost floor.
