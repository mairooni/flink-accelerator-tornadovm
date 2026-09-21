# Deployment

How the four pieces fit together, in what order they have to be built, and which
couplings between them fail at runtime rather than at build time.

## The four pieces

| piece | provides | where it ends up |
|---|---|---|
| **Flink**, `gpu-offload` branch | the accelerator IR, the SPI *interface*, and the planner seams that decide what is offloadable | the distribution |
| **TornadoVM**, CUDA backend | `TaskGraph`, the device runtime, and the library bindings (`tornado-cudf`, `tornado-cublas`, …) | its own SDK directory, referenced by path |
| **this repository** | the SPI *implementation*: kernel generation, staging, the operators and engines | one jar in `$FLINK_HOME/lib/` |
| **the examples module** | one benchmark or example class per jar | a jar passed to `bin/flink run` |

The shape worth understanding first: **Flink ships the interface and this
repository ships the implementation, and Flink does not depend on TornadoVM at
all.** `AcceleratorProvider` lives in `flink-table-runtime`;
`TornadoVmAcceleratorProvider implements AcceleratorProvider` lives here. That is
why Flink's own test suite can be green while never touching a device, and why a
TaskManager with no provider jar, no device or no TornadoVM flags runs the
code-generated CPU operator and logs the reason instead of failing.

## Build order, which is strict

```
Flink  ──mvn install──▶  ~/.m2  ──┐
                                  ├──▶  this repository  ──▶  provider jar
TornadoVM ──make BACKEND=cuda──▶ ─┘
```

Every dependency this repository declares on Flink and TornadoVM is
`provided` scope: it compiles against them and bundles neither, so both have to
be installed in the local Maven repository first.

```bash
# 1. Flink, which installs the SPI and the IR into ~/.m2
cd flink && ./mvnw install -DskipTests -Dfast

# 2. TornadoVM, which installs tornado-api, tornado-cublas, tornado-cudf …
cd TornadoVM && make BACKEND=cuda

# 3. this repository
mvn install
```

Change the SPI in Flink and Flink must be reinstalled before this builds. Change
TornadoVM's version and `<tornado.version>` here has to match — the pom currently
pins `6.1.1-jdk21-dev` against Flink `2.3.0`.

**JDK 21.** TornadoVM's off-heap arrays are built on `java.lang.foreign`, preview
on 21 and final on 22, and Flink 2.3 has no JDK 22+ profile — so 21 with preview
enabled is what a TaskManager running this actually runs on.

## Deploy

```bash
TORNADOVM_HOME=/path/to/tornadovm-6.1.1-jdk21-dev-cuda \
  flink-accelerator-tornadovm/scripts/gpu-cluster-setup.sh $FLINK_HOME
```

Four things happen, and each matters:

1. **The provider jar is copied into `lib/`**, putting it on the TaskManager
   classpath where `ServiceLoader` will find it.
2. **TornadoVM's JVM flags are appended to `conf/config.yaml`** under
   `env: java: opts: all:`. They are read from TornadoVM's own
   `tornado-argfile.template`, which is generated from `tornado --printJavaFlags`,
   so they stay correct across TornadoVM versions rather than being copied by
   hand. They carry `--add-modules` for the backend and library modules and
   `--enable-native-access=tornado.runtime`. They must reach **all three** JVMs —
   the client builds the plan, the JobManager holds it, the TaskManager runs the
   kernel — which is why they go in the `all` key.
3. **`flink-sql-parquet` is installed**, which the distribution does not ship and
   the columnar path needs.
4. **Memory and the GPU resource are sized**:
   `taskmanager.memory.task.off-heap.size=512m`, process size 16g, and
   `external-resource.gpu.amount=1`.

Overridable by environment variable: `PROVIDER_JAR`, `PARQUET_JAR`, `FLINK_SRC`,
`OFF_HEAP`, `PROCESS_MEM`, `GPU_RESOURCE_NAME`, `GPU_RESOURCE_AMOUNT`,
`DECLARE_GPU`.

**TornadoVM itself is not copied.** `TORNADOVM_HOME` is baked into the flags as a
path, so the SDK stays where it is and the cluster references it.

> **Re-run this script after every distribution build.** The assembly regenerates
> `config.yaml` and silently takes TornadoVM off the module path.

### Off-heap sizing is the trap that costs the most time

TornadoVM stages through `java.lang.foreign`, which counts against the JVM
direct-memory limit Flink derives from
`taskmanager.memory.task.off-heap.size` — zero by default. Raising it also means
raising the process size, because managed and network memory are fractions of the
total. A TaskManager that fails this check reports it **only in its `.out` file**,
so the symptom is a cluster with no workers and a job hanging on a slot.

## Run

The examples build one jar per class, and **nothing copies them into the
distribution** — that step is yours, and the `run-*.sh` helpers expect them in
`examples/table/`:

```bash
cp flink-accelerator-tornadovm-examples/target/HaversineBenchmark.jar \
   $FLINK_HOME/examples/table/
$FLINK_HOME/bin/start-cluster.sh
$FLINK_HOME/bin/flink run $FLINK_HOME/examples/table/HaversineBenchmark.jar --data /tmp/points
```

`flink-accelerator-tornadovm/scripts/` has a `run-<benchmark>.sh` per benchmark,
which sweeps the CPU and device arms and prints one line per run. Each one fails
loudly if its jar is not where it expects it.

A query reaches the device in four steps:

1. The user writes **ordinary SQL** and the deployment sets
   `table.exec.accelerator.enabled = true`. This is a deployment setting, never
   query syntax: no hint, annotation or per-query option is required, which is the
   project's transparency invariant.
2. The **planner** decides eligibility and substitutes an accelerated ExecNode. It
   knows nothing about GPUs — only about the IR.
3. On the TaskManager, `AcceleratorProviders` runs `ServiceLoader.load(...)`,
   finds this provider through
   `META-INF/services/org.apache.flink.table.runtime.accelerator.AcceleratorProvider`,
   and offers it the IR subtree.
4. The provider **accepts or declines**. A decline falls back to the generated CPU
   operator, which is why this is safe to switch on.

The log is the only evidence that counts:

```bash
grep -E "Accelerated on this TaskManager|Accelerator declined" \
  $FLINK_HOME/log/flink-*-taskexecutor-*.log
```

## The cuDF operators need RAPIDS libcudf

`GROUP BY`, `ORDER BY`, the join and the filter go through `tornado-cudf`, which
binds a shim over RAPIDS libcudf. Needed at **build** time to compile the shim:

```bash
python3 -m venv /tmp/cudfenv
/tmp/cudfenv/bin/pip install --extra-index-url=https://pypi.nvidia.com libcudf-cu12
export CUDF_HOME=$(echo /tmp/cudfenv/lib/python3.*/site-packages)/libcudf
make BACKEND=cuda     # in TornadoVM; skips the shim with a warning if absent
```

and at **run** time, where **every** RAPIDS directory has to be on
`LD_LIBRARY_PATH`, not just libcudf — `ldd` reports the shim as fine and `dlopen`
still fails without `rapids_logger` and `libnvcomp`:

```bash
SP=$(echo /tmp/cudfenv/lib/python3.*/site-packages)
export LD_LIBRARY_PATH=$SP/libcudf/lib64:$SP/librmm/lib64:$SP/libkvikio/lib64\
:$SP/rapids_logger/lib64:$SP/nvidia/libnvcomp/lib64:$SP/nvidia/cufile/lib:$LD_LIBRARY_PATH
```

Without it the binding reports itself unavailable and the cuDF-backed operators
decline cleanly — the same thing cuSPARSE does on a host with no CUDA toolkit.

## Couplings that fail at runtime, not at build time

These are the ones that have actually cost time.

**An engine class change needs the provider reinstalled.** An example jar carries
only its own class; every engine (`KMeansEngine`, `LogisticRegressionEngine`,
`GpuGramEngine`) comes from the provider jar in `lib/`. Edit one and re-run
`gpu-cluster-setup.sh`, or the cluster keeps the old class: a new class gives
`NoClassDefFoundError`, a new method gives `NoSuchMethodError`. The dangerous part
is the timing — the job dies in about 2 s where a working device arm takes 4 to 5,
so a benchmark that discards output reports the failure as an *improvement*.
Always confirm the harness printed a result line, and grep the TaskManager log for
the operator's own line.

**The provider jar and the TornadoVM SDK must be the same version.** The provider
calls `Cudf.sortedOrder(n, keys, order)`; an SDK carrying the older four-argument
form gives `NoSuchMethodError` at runtime. Nothing catches this at build time,
because both are `provided` scope.

**A planner or IR change has to reach the cluster through three jars**, in order:

```bash
./mvnw -o clean install -DskipTests -Dfast -pl flink-table/flink-table-planner
./mvnw -o clean install -DskipTests -Dfast \
  -pl flink-table/flink-table-planner-loader-bundle,flink-table/flink-table-planner-loader
./mvnw -o clean install -DskipTests -Dfast -pl flink-table/flink-table-api-java-uber
```

The loader bundles `flink-table-planner-loader-bundle`, and the IR ships inside
`flink-table-api-java-uber`. Miss one and the job runs, returns the right answer,
and measures the CPU in both arms. Verify by extracting the class from the shipped
jar and looking for the method you added:

```bash
unzip -p $FLINK_HOME/lib/flink-table-planner-loader-2.3.0.jar flink-table-planner.jar > /tmp/p.jar
unzip -o -q /tmp/p.jar '<your class>.class' && javap -c -p '<your class>.class' | grep '<your method>'
```

**Maven never deletes output for a deleted source.** `target/classes` accumulates
classes whose `.java` is gone, the jar plugin packages them, and shade carries
them downstream. After deleting or moving a source, build that module with `clean`
once.

**Do not `rm log/*` under a running cluster.** The TaskManager keeps writing to
deleted descriptors and the accelerator's decision lines live only there. Stop the
cluster first.

## What is not deployable yet

- **A clean checkout cannot build this**, because TornadoVM is not on Maven
  Central and the accelerator SPI is in no Flink release.
- **`require-device` and the cardinality gate are mutually exclusive.**
  AdaptiveBatch flattens the resource profile to `UNKNOWN`, so requiring a device
  needs the Default scheduler, which disables the cardinality gate.
- **`BatchExecAdaptiveJoin` is not offloaded**, and it is Flink 2.3's default plan
  for a shuffle join — so an out-of-the-box deployment never offloads a join
  unless it sets `table.optimizer.adaptive-broadcast-join.strategy` to `NONE`.
