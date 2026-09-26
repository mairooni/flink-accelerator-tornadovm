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
-->

# Running the two LLM pipeline experiments

Two Flink applications do the same job — screen eight million telemetry readings for anomalies,
roll the survivors up per machine, and have a language model write the maintenance note — and
differ in two things:

| | preprocessing | inference |
|---|---|---|
| **`GpuTriagePipeline`** | offloaded to the GPU through the TornadoVM accelerator | **jitllm, in the TaskManager's own JVM** |
| **`CpuTriagePipeline`** | Flink's code-generated operator, on the cores | **llama.cpp**, in a separate process, over HTTP |

The SQL text is identical in both. The measured result and what it means is in
`flink-accelerator-tornadovm/FINDINGS.md`, under *A Flink query and a language model on one GPU*.

---

## 1. What you need first

| | |
|---|---|
| an NVIDIA GPU | with enough VRAM for a 1.4 GiB model plus ~1 GiB of Flink staging. 8 GiB is comfortable. |
| **JDK 21** | the same one TornadoVM's SDK is built with. TornadoVM's off-heap arrays are `java.lang.foreign`, preview on 21, and the cluster must run the JDK the SDK was built against. |
| a CUDA toolkit | with `nvcc`. If it is CUDA 13 it may have no cuBLAS — the setup script handles that; see *Troubleshooting*. |
| **Maven** and **CMake** | for the Java and the llama.cpp builds. |
| four checkouts | Flink on `gpu-offload`, TornadoVM, this repository, jitllm, plus llama.cpp. |
| a built Flink distribution | `mvn install -DskipTests` in the Flink checkout, so the accelerator SPI is in `~/.m2` and `flink-dist/target/.../flink-2.3.0` exists. This is the long one. |
| a GGUF model | the report uses `Qwen3-0.6B-f16.gguf`, which ships in the jitllm checkout. Both engines load the same file. |
| **Hadoop jars** | the input is Parquet and `flink-sql-parquet` bundles `org.apache.parquet.*` but none of `org.apache.hadoop.*`. Either have `hadoop` on the PATH or export `HADOOP_CLASSPATH` yourself. |

The scripts find everything through environment variables with defaults. Override any that do
not match your machine:

```bash
export FLINK_SRC=~/Projects/flink                       # the gpu-offload checkout
export FLINK_HOME=$FLINK_SRC/flink-dist/target/flink-2.3.0-bin/flink-2.3.0
export TORNADOVM_SRC=~/Projects/TornadoVM
export JITLLM_SRC=~/Projects/GPULlama3-Beehive/GPULlama3.java
export LLAMACPP_SRC=~/Projects/llama.cpp
export MODEL=$JITLLM_SRC/Qwen3-0.6B-f16.gguf
export JDK21=~/Projects/JDKs/jdk-21.0.3
export WORK=~/gpu-bench-data/flink-llm                  # dataset, CUDA shim, env file
```

### One patch that is not upstream

The accelerated arm needs the TornadoVM branch **`fix/device-reset-scoped-to-execution-plan`**.
Without it the model dies on the *second* query with

```
[ERROR] reset() was called after warmup() on device: [NVIDIA CUDA] -- ...
```

because freeing the query's execution plan at the end of a job marked the whole device as reset,
and the resident model's warmed-up plan is on that device too. Check the branch out before
running the setup script. A single cold run works without it; nothing warm does.

---

## 2. Set the environment up — once

```bash
cd <this repository>
./flink-accelerator-tornadovm/scripts/llm-bench-setup.sh
```

It does seven things, in an order that is not a preference — each step installs the artifact the
next one compiles against, and out of order the build quietly succeeds against a stale jar and
fails at run time instead:

1. **checks prerequisites** — JDK, checkouts, model, GPU — before building anything, and refuses
   a TornadoVM SDK that does not report a `CUDADriver` (an OpenCL SDK on the PATH will run
   everything and report nothing wrong, on the wrong device);
2. **builds TornadoVM** (`make BACKEND=cuda`), which installs `tornado-*` into `~/.m2`;
3. **builds jitllm** against that exact TornadoVM version;
4. **builds llama.cpp** with CUDA, assembling a toolkit shim first if cuBLAS is missing;
5. **builds this repository** — the provider and one runnable jar per example;
6. **deploys into the distribution** — the provider jar and TornadoVM's JVM flags via
   `gpu-cluster-setup.sh`, then jitllm into `lib/` and the properties its batched-prefill path is
   gated on via `llm-pipeline-setup.sh`, then raises the TaskManager's task off-heap memory;
7. **generates the dataset** — 8,000,000 rows of Parquet, read by both arms, about 5 s.

Finally it writes `$WORK/llm-bench.env`, which the run script sources. Every step is skippable
when you are iterating:

```bash
./llm-bench-setup.sh --skip-tornadovm --skip-llamacpp    # only rebuild the Java side
./llm-bench-setup.sh --skip-data --rows 2000000          # a smaller dataset
```

### Why jitllm goes into `lib/` and not into the job jar

A model kept resident across jobs has to be held by a class the **parent** classloader owns.
Flink discards the user-code classloader when a job finishes, and every static field it defined
goes with it — so a model held from the job jar is collected between queries and the warm case
cannot exist. `ResidentEngines` keeps it in a map reachable from `System.getProperties()`, which
a bootstrap class owns, and that is only safe because jitllm's own classes come from `lib/`.

---

## 3. Run one experiment

Start the cluster, run once, stop the cluster:

```bash
./flink-accelerator-tornadovm/scripts/llm-bench-run.sh --gpu     # GPU preprocess + jitllm
./flink-accelerator-tornadovm/scripts/llm-bench-run.sh --cpu     # CPU preprocess + llama.cpp
```

| flag | |
|---|---|
| `--gpu` / `--cpu` | which arm. Required. |
| `--warm` | run once untimed to make the engine resident, then time a second run. **See the warning below.** |
| `--modes N` | operating modes per reading, i.e. arithmetic per row. Default 8. |
| `--explain` | also print the plan and the accelerator's accept/decline reasons. |

The cluster is torn down on every exit path, including failure, so a crashed run does not leave a
TaskManager or a `llama-server` holding VRAM.

### Cold and warm are different experiments, and the difference is the whole result

A plain `--gpu` run is **cold**: it starts a cluster, and the first query loads 1.4 GiB of weights
and JIT-compiles the model's kernels. Expect roughly:

| | cold (`--gpu` / `--cpu`) | warm (`--warm`) |
|---|---:|---:|
| GPU preprocess + jitllm | 15 – 24 s | **~7.7 s** |
| CPU preprocess + llama.cpp | 15 – 18 s | ~14.1 s |

**Cold, the two arms are level.** The accelerated arm saves about 7.6 s on preprocessing and
spends all of it loading the model. The 1.84x in the report is the warm number, which is what a
long-lived TaskManager actually serves — it loads the model once and answers thousands of
queries. If you run each arm once and conclude they are the same, that is the reason.

Cold timings also vary by several seconds with page cache: a 1.4 GiB GGUF that is already cached
loads in about a second, and one that is not takes three.

### What the output tells you

```
---- GPU preprocessing + jitllm (in the TaskManager's JVM) ----
  engine=jitllm
  model_load_ms=0.7          <- 0 means it was already resident; thousands means it loaded now
  resident=true              <- the warm/cold flag, and the first thing to check
  prompt_tokens=1657
  generated_tokens=255
  prefill_ms=946.5           <- reading the prompt
  decode_ms=2063.9           <- writing the answer
  inference_wall_ms=3015.7   <- minus the two above = the cost of reaching the engine
  decode_tok_s=123.55
  flink run wall time: 8.06 s

---- what the TaskManager decided ----
  Accelerated on this TaskManager: provider tornadovm claims 15.15x over CPU
  batches=31  rows_in=8000000  rows_out=2908 (0.0% selectivity)
  ...
```

That last block matters more than it looks. **The offload decision is not in the plan.** The
planner marks a subtree *eligible* and ships both an accelerated and a code-generated operator;
the TaskManager picks between them when the task is deployed, because that is the first place the
device is observable. A green run proves nothing about which one executed — those lines do.

If instead you see

```
Accelerator declined on this TaskManager, running the generated CPU operator: ...
```

the GPU arm ran on the CPU and the comparison is meaningless. See *Troubleshooting*.

---

## 4. Reproducing the individual numbers in the report

Everything below assumes `source $WORK/llm-bench.env` and a running cluster
(`$FLINK_HOME/bin/start-cluster.sh`). `J=$LLM_BENCH_JAR_PREFIX`.

**The end-to-end comparison, five interleaved repetitions.** Interleave the arms rather than
running five of one then five of the other: the card's clock falls as it warms — 2430 MHz
observed against a 3105 MHz maximum — so a block design gives whichever arm goes first a colder
card. The 20 s pause lets it settle.

```bash
for r in 1 2 3 4 5; do
  for arm in Gpu Cpu; do
    s=$(date +%s.%N)
    $FLINK_HOME/bin/flink run -c org.apache.flink.table.examples.java.gpu.llm.${arm}TriagePipeline \
        $J-${arm}TriagePipeline.jar --data $LLM_BENCH_DATA --model $LLM_BENCH_MODEL >/dev/null 2>&1
    echo "$arm $(python3 -c "print(f'{$(date +%s.%N)-$s:.2f}')")"
    sleep 20
  done
done
```

**Preprocessing alone**, with no model loaded at all. `--digest-only` replaces the model call
with a checksum of the digest. Time it with the JobManager's own job duration rather than the
client's wall time: the client's includes Calcite planning a query whose `LEAST` over eight modes
is a large expression, and that grows with `--modes` identically in both arms.

```bash
OUT=$($FLINK_HOME/bin/flink run -c ...GpuTriagePipeline $J-GpuTriagePipeline.jar --digest-only 2>&1)
JID=$(echo "$OUT" | grep -oP 'JobID \K[0-9a-f]+')
curl -s localhost:8081/jobs/$JID | python3 -c 'import json,sys; print(json.load(sys.stdin)["duration"], "ms")'
```

Both arms must print `+I[48, 2908, 1128, 192.59903094408028]`. Identical to the last bit is the
correctness gate, not a coincidence: every operation in the score is correctly rounded in IEEE
754, so a device and a CPU have to agree.

**The device-side breakdown** is in the TaskManager log after any GPU run:

```bash
grep -a -A12 "GpuCalcOperator GpuCalcSpec" \
     $FLINK_HOME/log/flink-$(whoami)-taskexecutor-0-$(hostname).log | tail -13
```

**The arithmetic-intensity sweep.** Above 8 modes the provider declines the subtree — the ceiling
is 1024 operations a row — so raise it on the TaskManager first, and check the digests still
match afterwards:

```bash
#   add to env.java.opts.all in $FLINK_HOME/conf/config.yaml, then restart:
#   -Dflink.accelerator.tornadovm.maxOpsPerRow=16384
for m in 8 16 32; do
  for arm in Gpu Cpu; do ... --digest-only --modes $m ; done
done
```

**The engines on their own**, with the cluster stopped:

```bash
cd $JITLLM_SRC   && ./jitllm --gpu --model $MODEL --bench --bench-args "-p 256 -n 64 -r 2 -b 256"
cd $LLAMACPP_SRC && ./build/bin/llama-bench -m $MODEL -p 256 -n 64 -ngl 99 -r 2
```

`-b 256` is not optional for a fair reading of jitllm. Without it prompt processing runs one
token per forward — 138 tok/s instead of 5422 — because the batched-prefill MMA path is gated on
system properties read when the state buffers are allocated.

---

## 5. Troubleshooting

**The GPU arm says the accelerator declined.** Run with `--explain` and read the `GPU Offload`
section, which names the reason per subtree.

- *`not expressible on device: computes a value a device may round differently`* — an expression
  reached a filter, grouping or sort through a function whose device result may drift. `SQRT` is
  the common one: Calcite rewrites it to `POWER`. The shipped score avoids it by working in
  squared distance.
- *`N operations a row exceeds the ceiling of 1024`* — raise
  `-Dflink.accelerator.tornadovm.maxOpsPerRow`. It is a correctness guard set from an OpenCL
  measurement, and on CUDA it is conservative; verify the digests still match after raising it.
- *`no accelerator provider on this TaskManager's classpath`* — `gpu-cluster-setup.sh` was not run,
  or a `flink-dist` build regenerated `conf/config.yaml` afterwards and took the TornadoVM flags
  out with it. Re-run it; `grep -c tornado.cudf conf/config.yaml` should not be 0.

**`reset() was called after warmup()`** — the TornadoVM branch above is not checked out.

**`OutOfMemoryError: Direct buffer memory` inside `TornadoWorkspaces`** — the engine's device
workspaces are direct buffers and Flink charges them against a budget sized from its own memory
model. Raise `taskmanager.memory.task.off-heap.size`; the setup script sets it to 4g.

**`ClassNotFoundException: org.apache.hadoop.conf.Configuration`** — `HADOOP_CLASSPATH` is not set
for the client. It must be exported in the shell that runs `flink run`, not only the cluster.

**TornadoVM's build fails in `cudnn-jni` or `cutlass-jni`** at cmake's CUDA compiler
identification. CUDA 13.3 refuses GCC 16, and CUTLASS needs two further flags. Put a wrapper
`nvcc` first on the PATH that adds `-allow-unsupported-compiler` and points `find_library` at a
toolkit that has cuBLAS and cuDNN.

**`make fast-tests` reports failures in `unittests.tile.*`.** On a machine with a pip-installed
CUDA toolkit under `~/.local/lib/python3.*/site-packages/nvidia/`, the runtime tile compiler
prefers that `nvcc` and its include tree has no `nv/target`. 63 non-whitelisted failures, all
pre-existing, none related to this work. Baseline with your change stashed before believing it.

**The two arms take the same time.** Check `resident=` in the output. Cold, they are level, and
that is the honest result — see the table above.

**The job appears to hang while generating data.** `datagen` defaults to 10,000 rows a second,
which makes 8M rows a fourteen-minute sleep. The example sets `rows-per-second` high; if you have
written your own generator, set it.
