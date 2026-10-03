# Three demos: Flink SQL on a GPU, through TornadoVM

Everything here runs ordinary Flink SQL. No demo uses a hint, an annotation, a
UDF or any syntax a device could be inferred from — the only thing a query
author writes that they would not otherwise write is `NOT NULL` in the DDL.

| | what it shows | the library |
|---|---|---|
| `demo-haversine.sh` | a SQL expression compiled to a CUDA kernel | none — this is the JIT half |
| `demo-regex.sh` | a whole SQL subtree served by a CUDA library | cuDF: `read_parquet` + `contains_re` |
| `demo-llm.sh` | GPU-preprocessed SQL feeding a resident language model | cuDF, and cuDNN inside jitllm |

## Three scripts

| | | when |
|---|---|---|
| `./1-fetch.sh` | clones TornadoVM, Flink, jitllm and llama.cpp at the right branches, builds them and this repository, deploys the provider into the Flink distribution | once, on a new machine |
| `./2-generate-data.sh` | the log corpora (~2.4 GB), demo 3's model (1.44 GiB), and demo 3's own setup | once, on a new machine |
| `source ./3-env.sh` | sets every path the demos need, and says what is missing | every shell |

A fresh laptop runs all three; a machine that is already set up runs only the
third. Both of the first two are idempotent, so a failed step can be fixed and
the script re-run.

```bash
git clone <this repo> && cd flink-accelerator-tornadovm/demos
./1-fetch.sh            # 40-70 minutes cold; Flink's own build is most of it
./2-generate-data.sh    # a few minutes
source ./3-env.sh
```

This repository is not fetched by `1-fetch.sh` — the script ships inside it, so
it is already the clone you are standing in, and it is only built. The other two
are left alone if they are already checked out on a different branch or have
local changes: a setup script has no business switching your branch.

`1-fetch.sh` needs, and checks for: an NVIDIA GPU, a CUDA toolkit whose nvcc
accepts the host compiler, **JDK 21**, Maven, CMake, Python 3, git and a C++20
compiler. Override any path by exporting it first — `DEMO_ROOT`,
`TORNADOVM_SRC`, `FLINK_SRC`, `PROVIDER_SRC`, `DATA_ROOT`, `RAPIDS_HOME`,
`JAVA_HOME`, `CUDA_PATH` — or, on a machine whose checkouts live elsewhere, put
them in `demos/env.local.sh`, which `3-env.sh` reads and which is not committed.

Then, in a shell where `3-env.sh` has been sourced:

```bash
./demo-haversine.sh --print-kernel     # 8M rows; the CUDA lands in the TaskManager .out
./demo-haversine.sh --keep-cluster     # leave the cluster up for the web UI
./demo-regex.sh --reuse-cluster        # run into the cluster that is already up,
                                       # so the UI keeps the earlier jobs too
./stop.sh                              # stop a cluster left up that way

./demo-regex.sh --rows 1 --patterns 8 --print-bytecodes   # library tasks and the
                      # generated kernel as nodes of one task graph. --rows 1 is the
                      # 1M corpus: two Parquet files, so ~130 bytecode lines not 2,000
./demo-regex.sh --patterns 8
./demo-regex.sh --patterns 1           # below the floor: the region declines
./demo-llm.sh
```

Each script starts a cluster, runs one job, prints what the planner and the
accelerator decided, and stops the cluster on every exit path. Flags combine in
any order, so `--print-kernel --keep-cluster` does both. With `--keep-cluster`
the cluster stays up afterwards — the job is then visible at
<http://localhost:8081> — and `./stop.sh` shuts it down.

### Demo 3

Nothing extra to run. `1-fetch.sh` clones jitllm (`beehive-lab/jitllm`) and
llama.cpp with the rest; `2-generate-data.sh` downloads the model and then
finishes the setup — building both engines and the triage dataset — because
`llm-bench-setup.sh` refuses to start without the model and the model is
fetched there. The first run adds about fifteen minutes for the two engines.

`*.gguf` is gitignored in the jitllm checkout, so the clone brings no weights;
the file comes from Hugging Face (`gvij/qwen3-0.6b-gguf`, Qwen3-0.6B fp16) and
is checked for the GGUF magic, because a truncated download fails deep inside
the engine's loader rather than at startup.

Skip all of it with `./1-fetch.sh --skip-llm` and
`./2-generate-data.sh --skip-model`; add it to an existing setup with
`./1-fetch.sh --only-llm` then `./2-generate-data.sh`. Point `MODEL` at a GGUF
you already have, or `MODEL_URL` at a different one.

### By talk

The two decks need different subsets, and each is a flag away:

| | demos | corpora |
|---|---|---|
| lab talk | haversine, LLM | `haversine-8000000`, the triage readings |
| libraries talk | regex | `logs1` for bytecodes, `logs16` or `logs64` for timing |

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
