/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.table.gpu.metrics;

import java.io.IOException;
import java.io.InputStream;
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.zip.CRC32;

/**
 * Everything a reported speedup needs beside the number, collected rather than remembered.
 *
 * <h2>Why this exists</h2>
 *
 * <p>This project's numbers have moved 6% between sittings on the same machine and the same code. A
 * figure with no environment attached cannot be compared with a figure taken a month later, and
 * cannot be reproduced by anyone else at all — which makes it an anecdote rather than a result. The
 * review is explicit about what has to accompany one: exact commits and build mode, JVM flags,
 * CUDA/driver versions, GPU and CPU model, clocks and power mode, host load, dataset checksum and
 * shape, parallelism and batch size, warm-up policy, repetitions, and a distribution rather than a
 * single time.
 *
 * <p>Collected at run time rather than written down, because a hand-maintained header is a header
 * that is wrong the first time something changes. Everything here is read from the machine the
 * benchmark is running on.
 *
 * <h2>The distribution, not the mean</h2>
 *
 * <p>{@link #summary} reports median, p95, min, max and standard deviation over the measured runs,
 * and reports the first run separately. The first run pays JIT compilation, kernel compilation and
 * page faults, and folding it into a mean produces a number that describes neither a cold start nor
 * a warm one. Which runs count as warm-up is stated rather than assumed.
 */
public final class BenchmarkRun {

    private final String label;
    private final int warmUp;
    private final List<Long> nanos = new ArrayList<>();

    /**
     * @param warmUp how many leading runs are excluded from the statistics. They are still
     *     reported: a cold run that is thirty times a warm one is a fact about the deployment, not
     *     noise to be dropped quietly.
     */
    public BenchmarkRun(String label, int warmUp) {
        this.label = label;
        this.warmUp = Math.max(0, warmUp);
    }

    public void record(long elapsedNanos) {
        nanos.add(elapsedNanos);
    }

    /** How many runs have been recorded, warm-up included. */
    public int runs() {
        return nanos.size();
    }

    // ---------------------------------------------------------------------------------------
    // The environment, which is the half that cannot be recovered afterwards
    // ---------------------------------------------------------------------------------------

    /**
     * Every fact about this machine and this build that a reader would have to ask for.
     *
     * <p>Missing values are printed as such rather than omitted. "driver: unavailable" says the
     * benchmark ran somewhere without {@code nvidia-smi}, which is itself worth knowing; a line
     * that simply is not there says nothing at all.
     */
    public static String environment() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== environment ===\n");

        sb.append(line("host", hostname() + "  " + System.getProperty("os.name") + " " + kernel()));
        sb.append(
                line(
                        "cpu",
                        cpuModel()
                                + "  ("
                                + Runtime.getRuntime().availableProcessors()
                                + " logical)"));
        sb.append(
                line(
                        "load",
                        read("/proc/loadavg")
                                .map(s -> s.split("\\s+"))
                                .map(
                                        p ->
                                                p.length >= 3
                                                        ? p[0] + " " + p[1] + " " + p[2]
                                                        : "unavailable")
                                .orElse("unavailable")));
        sb.append(line("gpu", nvidiaSmi("name,driver_version,memory.total")));
        sb.append(line("power/clocks", nvidiaSmi("power.limit,clocks.max.sm,clocks.max.mem")));
        sb.append(line("cuda", cudaVersion()));
        sb.append(line("rapids/cudf", cudfLibrary()));

        sb.append(
                line(
                        "jvm",
                        System.getProperty("java.vendor")
                                + " "
                                + System.getProperty("java.version")
                                + "  "
                                + System.getProperty("java.vm.name")));
        sb.append(line("jvm flags", jvmFlags()));
        sb.append(line("heap max", (Runtime.getRuntime().maxMemory() >> 20) + " MiB"));

        sb.append(line("provider", implementationVersion(BenchmarkRun.class)));
        sb.append(line("tornadovm", tornadoVersion()));
        sb.append(line("git", gitDescription()));
        return sb.toString();
    }

    /**
     * What the input actually is, including a checksum, so two runs can be shown to have read the
     * same bytes.
     *
     * <p>A row count is not enough. The same generator run twice produces the same count and
     * different values, and a benchmark whose input changed between arms is comparing two things.
     * CRC32 over the file contents in a stable order is cheap — a few hundred MiB take about a
     * second — and it is a claim anyone can re-check.
     */
    public static String dataset(String path) {
        Path root = Paths.get(path);
        if (!Files.exists(root)) {
            return line("dataset", path + "  (absent)");
        }
        try (Stream<Path> walk = Files.walk(root)) {
            List<Path> files =
                    walk.filter(Files::isRegularFile).sorted().collect(Collectors.toList());
            long bytes = 0;
            CRC32 crc = new CRC32();
            byte[] buffer = new byte[1 << 16];
            for (Path file : files) {
                crc.update(root.relativize(file).toString().getBytes("UTF-8"));
                try (InputStream in = Files.newInputStream(file)) {
                    int read;
                    while ((read = in.read(buffer)) > 0) {
                        crc.update(buffer, 0, read);
                        bytes += read;
                    }
                }
            }
            return line(
                    "dataset",
                    String.format(
                            "%s  %d files  %.1f MiB  crc32=%08x",
                            path, files.size(), bytes / (1024.0 * 1024.0), crc.getValue()));
        } catch (IOException | RuntimeException unreadable) {
            return line("dataset", path + "  (unreadable: " + unreadable + ")");
        }
    }

    /** The shape of the job, which the benchmark knows and the machine does not. */
    public static String job(int parallelism, int batchSize, long rows, String extra) {
        return line(
                "job",
                String.format(
                        "parallelism=%s  batch=%d  rows=%d%s",
                        parallelism > 0 ? Integer.toString(parallelism) : "(default)",
                        batchSize,
                        rows,
                        extra == null || extra.isEmpty() ? "" : "  " + extra));
    }

    // ---------------------------------------------------------------------------------------
    // The distribution
    // ---------------------------------------------------------------------------------------

    public String summary() {
        if (nanos.isEmpty()) {
            return "=== " + label + " ===\n  no runs recorded\n";
        }
        List<Long> measured = nanos.subList(Math.min(warmUp, nanos.size()), nanos.size());
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%n=== %s ===%n", label));
        sb.append(
                String.format(
                        "  runs=%d  warm-up=%d (excluded)  measured=%d%n",
                        nanos.size(), Math.min(warmUp, nanos.size()), measured.size()));
        sb.append(String.format("  first (cold) %10.1f ms%n", nanos.get(0) / 1e6));
        if (measured.isEmpty()) {
            sb.append("  every run was warm-up; nothing to summarise\n");
            return sb.toString();
        }
        List<Long> sorted = new ArrayList<>(measured);
        sorted.sort(Comparator.naturalOrder());
        double mean = measured.stream().mapToLong(Long::longValue).average().orElse(0);
        double variance =
                measured.stream().mapToDouble(n -> (n - mean) * (n - mean)).sum()
                        / Math.max(1, measured.size());
        sb.append(String.format("  median       %10.1f ms%n", percentile(sorted, 50) / 1e6));
        sb.append(String.format("  p95          %10.1f ms%n", percentile(sorted, 95) / 1e6));
        sb.append(String.format("  min          %10.1f ms%n", sorted.get(0) / 1e6));
        sb.append(String.format("  max          %10.1f ms%n", sorted.get(sorted.size() - 1) / 1e6));
        sb.append(String.format("  mean         %10.1f ms%n", mean / 1e6));
        sb.append(
                String.format(
                        "  stddev       %10.1f ms  (%.1f%% of the median)%n",
                        Math.sqrt(variance) / 1e6,
                        100.0 * Math.sqrt(variance) / Math.max(1.0, percentile(sorted, 50))));
        return sb.toString();
    }

    /**
     * Nearest-rank, which is the one that needs no interpolation and no apology at small n.
     *
     * <p>A p95 over ten runs is the highest of them whichever definition is used; saying which one
     * was meant still matters, because a reader comparing two projects' p95 is otherwise comparing
     * two different statistics.
     */
    private static double percentile(List<Long> sorted, int percentile) {
        int rank = (int) Math.ceil(percentile / 100.0 * sorted.size());
        return sorted.get(Math.max(0, Math.min(sorted.size() - 1, rank - 1)));
    }

    // ---------------------------------------------------------------------------------------

    private static String line(String key, String value) {
        return String.format("  %-14s %s%n", key + ":", value);
    }

    private static String hostname() {
        String name = System.getenv("HOSTNAME");
        if (name != null && !name.isEmpty()) {
            return name;
        }
        return read("/proc/sys/kernel/hostname").orElse("unknown");
    }

    private static String kernel() {
        return read("/proc/sys/kernel/osrelease").orElse(System.getProperty("os.version"));
    }

    private static String cpuModel() {
        try (Stream<String> lines = Files.lines(Paths.get("/proc/cpuinfo"))) {
            return lines.filter(l -> l.startsWith("model name"))
                    .findFirst()
                    .map(l -> l.substring(l.indexOf(':') + 1).trim())
                    .orElse(System.getProperty("os.arch"));
        } catch (IOException | RuntimeException unavailable) {
            return System.getProperty("os.arch");
        }
    }

    private static String cudaVersion() {
        String fromSmi = command("nvidia-smi", "--query", "--display=COMPUTE");
        if (fromSmi != null) {
            for (String l : fromSmi.split("\n")) {
                if (l.contains("CUDA Version")) {
                    return l.substring(l.indexOf(':') + 1).trim();
                }
            }
        }
        String nvcc = command("nvcc", "--version");
        if (nvcc != null) {
            for (String l : nvcc.split("\n")) {
                if (l.contains("release")) {
                    return l.trim();
                }
            }
        }
        return "unavailable";
    }

    /** Which cuDF shim is actually loadable, which decides how much of the pipeline runs. */
    private static String cudfLibrary() {
        String sdk = System.getenv("TORNADOVM_HOME");
        if (sdk == null) {
            return "TORNADOVM_HOME unset";
        }
        Path shim = Paths.get(sdk, "lib", "libtornado-cudf.so");
        if (!Files.exists(shim)) {
            return "libtornado-cudf.so not built";
        }
        String ldd = command("ldd", shim.toString());
        if (ldd == null) {
            return shim + " (dependencies unchecked)";
        }
        List<String> missing =
                Arrays.stream(ldd.split("\n"))
                        .filter(l -> l.contains("not found"))
                        .map(l -> l.trim().split("\\s+")[0])
                        .collect(Collectors.toList());
        return missing.isEmpty()
                ? shim + " (all dependencies resolve)"
                : shim + " MISSING " + missing;
    }

    private static String nvidiaSmi(String fields) {
        String out = command("nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader");
        return out == null ? "unavailable" : out.trim().replace("\n", " | ");
    }

    /**
     * The flags the JVM was actually given, which is not the same as the flags someone meant.
     *
     * <p>{@code --enable-preview} and the module path are the difference between the provider
     * loading and the job silently running on the CPU, so they belong in the record of a run.
     */
    private static String jvmFlags() {
        try {
            List<String> args = ManagementFactory.getRuntimeMXBean().getInputArguments();
            if (args.isEmpty()) {
                return "(none)";
            }
            // A TornadoVM-enabled JVM is given about four kilobytes of --add-opens and
            // --add-exports, one line per module pair, and printing them entire buries the run
            // record in noise nobody reads. The ones kept are the ones that decide whether the
            // provider can load at all -- preview, JVMCI, the module and library paths, the heap
            // and GC -- and the count of what was dropped keeps the summary honest.
            List<String> interesting =
                    args.stream()
                            .filter(
                                    a ->
                                            !a.startsWith("--add-opens")
                                                    && !a.startsWith("--add-exports")
                                                    && !a.startsWith("-Dlog")
                                                    && !a.startsWith("-Dtornado.load"))
                            .collect(Collectors.toList());
            int dropped = args.size() - interesting.size();
            return String.join(" ", interesting)
                    + (dropped == 0 ? "" : "  [+" + dropped + " --add-opens/--add-exports/-Dlog]");
        } catch (RuntimeException unavailable) {
            return "unavailable";
        }
    }

    private static String tornadoVersion() {
        try {
            Class<?> api = Class.forName("uk.ac.manchester.tornado.api.TornadoExecutionPlan");
            return implementationVersion(api);
        } catch (Throwable absent) {
            return "not on the classpath";
        }
    }

    private static String implementationVersion(Class<?> type) {
        Package pkg = type.getPackage();
        String version = pkg == null ? null : pkg.getImplementationVersion();
        return version == null ? "(unversioned build)" : version;
    }

    /**
     * The commit, when the benchmark is run from a checkout.
     *
     * <p>It usually is not — a cluster run has a jar and no repository — so this is best effort and
     * says so. The jar's implementation version is the fallback, and neither being available is
     * itself a finding about how the run was made.
     */
    private static String gitDescription() {
        String described = command("git", "describe", "--always", "--dirty", "--long");
        return described == null ? "(not a checkout)" : described.trim();
    }

    private static java.util.Optional<String> read(String path) {
        try {
            return java.util.Optional.of(
                    new String(Files.readAllBytes(Paths.get(path)), "UTF-8").trim());
        } catch (IOException | RuntimeException unavailable) {
            return java.util.Optional.empty();
        }
    }

    /** Runs a command and returns its output, or null if it is absent or fails. */
    private static String command(String... argv) {
        try {
            Process process = new ProcessBuilder(argv).redirectErrorStream(true).start();
            String out;
            try (InputStream in = process.getInputStream()) {
                java.io.ByteArrayOutputStream buffer = new java.io.ByteArrayOutputStream();
                byte[] chunk = new byte[4096];
                int read;
                while ((read = in.read(chunk)) > 0) {
                    buffer.write(chunk, 0, read);
                }
                out = new String(buffer.toByteArray(), "UTF-8");
            }
            if (!process.waitFor(10, TimeUnit.SECONDS) || process.exitValue() != 0) {
                return null;
            }
            return out;
        } catch (IOException | InterruptedException | RuntimeException unavailable) {
            return null;
        }
    }
}
