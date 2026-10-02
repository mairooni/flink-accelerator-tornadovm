# Three demos: Flink SQL on a GPU, through TornadoVM

Everything here runs ordinary Flink SQL. No demo uses a hint, an annotation, a
UDF or any syntax a device could be inferred from — the only thing a query
author writes that they would not otherwise write is `NOT NULL` in the DDL.

| | what it shows | the library |
|---|---|---|
| `demo-haversine.sh` | a SQL expression compiled to a CUDA kernel | none — this is the JIT half |
| `demo-regex.sh` | a whole SQL subtree served by a CUDA library | cuDF: `read_parquet` + `contains_re` |
| `demo-llm.sh` | GPU-preprocessed SQL feeding a resident language model | cuDF, and cuDNN inside jitllm |

## Getting there from nothing

```bash
git clone <this repo> && cd flink-accelerator-tornadovm/demos
./setup.sh
```

`setup.sh` is idempotent — re-run it after fixing anything and it skips what is
already done. It clones TornadoVM, Flink and this repository, fetches the
RAPIDS libcudf wheels, builds all three, deploys the provider into the Flink
distribution, and generates the log corpus. Budget 40–70 minutes on a cold
machine; Flink's own build is most of it.

It needs, and checks for: an NVIDIA GPU, a CUDA toolkit, **JDK 21**, Maven,
CMake, Python 3, git and a C++20 compiler. Override any path by exporting it
first — `DEMO_ROOT`, `TORNADOVM_SRC`, `FLINK_SRC`, `PROVIDER_SRC`, `DATA_ROOT`,
`RAPIDS_HOME`, `JAVA_HOME`, `CUDA_PATH`.

Then:

```bash
./demo-haversine.sh --print-kernel     # and the CUDA it generated
./demo-regex.sh --patterns 8
./demo-regex.sh --patterns 1           # below the floor: the region declines
./demo-llm.sh
```

Each script starts a cluster, runs one job, prints what the planner and the
accelerator decided, and stops the cluster on every exit path.

## Things that will bite on a new machine

**The JDK must be the one TornadoVM was built with.** TornadoVM's off-heap
arrays are `java.lang.foreign`, preview on 21, so "a JDK 21" is not enough if
two are installed. `common.sh` resolves one and prints it.

**RAPIDS needs six directories on `LD_LIBRARY_PATH`, not three.** The last is
auditwheel's side-car and holds the only copy of `libzstd`. Omit any one and
cuDF reports itself unavailable with no diagnostic, which reads exactly like a
shim that was never built — the demo then runs on the CPU and prints the right
answer. `common.sh` sets all six.

**Parquet needs a real Hadoop client**, one consistent version of it.
`flink-sql-parquet` bundles `org.apache.parquet.*` and none of
`org.apache.hadoop.*`. A `find` over `~/.m2` collects two Hadoop versions and
three woodstox builds and fails at run time with a `NoSuchMethodError`;
`common.sh` resolves `hadoop-client:3.3.4` with Maven and caches it.

**cuDNN is optional but wanted.** The cuDNN and CUTLASS bindings live in their
own Maven profile, so a host without them still gets the core backend. Without
cuDNN the first two demos are unaffected and the LLM demo still runs, on a
prefill path about twenty times slower. `setup.sh` reports which it found.

**A green run proves nothing about where it ran.** The offload decision is
taken on the TaskManager, because that is the first place a device is
observable, so the plan does not record it. Each script prints the decision
line afterwards; read it.

## Showing the code

- `HaversineSQLExample.java` — the query, the three session settings, and a
  long comment on why this expression earns a device and `val * 2.0 + 1.0`
  does not.
- `GrokSQLExample.java` — the query and the pattern sets. The all-matching set
  is the default and is deliberate: patterns that reject rows let SQL's `AND`
  short-circuit, and the device has no equivalent, so measuring against an arm
  doing an eighth of the work would flatter this one. `--selective` is the
  opposite extreme.
- `GpuGrokSpec.java` — what the provider recognises, and `unsupported`, the
  list of regex constructs refused because cuDF and `java.util.regex` do not
  agree on them.
- `DeviceParquetGrokSource.java` — the region: one task graph, cuDF read, k
  library tasks, one generated kernel.
