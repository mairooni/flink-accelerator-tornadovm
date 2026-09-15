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
- **TornadoVM 6.0.1-jdk21-dev**, built locally. Its own build installs
  `tornado-api`, `tornado-annotation` and `tornado-cublas` into `~/.m2`.
- **JDK 21.** TornadoVM's off-heap arrays are built on `java.lang.foreign`, a
  preview API on 21 and final on 22; Flink 2.3 has no JDK 22+ profile, so
  21-with-preview is what a TaskManager running this actually runs on.

## Build

    mvn install

## Deploy

Put `flink-accelerator-tornadovm/target/flink-accelerator-tornadovm-*.jar` in a
distribution's `lib/`, and start the TaskManagers with TornadoVM's JVM arguments.
`flink-accelerator-tornadovm/scripts/gpu-cluster-setup.sh` does both.

A TaskManager without the jar, without a device, or without those JVM arguments
runs the code-generated operator instead and logs why. Nothing fails.

## Layout

| module | what it is |
|---|---|
| `flink-accelerator-tornadovm` | the provider: kernel generation, gathers, the staging operator |
| `flink-accelerator-tornadovm-examples` | runnable examples and benchmarks, one jar each |

`flink-accelerator-tornadovm/FINDINGS.md` is the measurement record: where the
time actually goes, and which of the original predictions it contradicted.
