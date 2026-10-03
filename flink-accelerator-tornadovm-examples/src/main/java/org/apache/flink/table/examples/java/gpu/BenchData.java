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

package org.apache.flink.table.examples.java.gpu;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Where the generated corpora live, when an example was not told.
 *
 * <p>Every example takes the directory as an argument, and the demo scripts always pass one. What
 * this fixes is the fallback: an absolute path baked into the jar is one machine's path, and on any
 * other machine it fails in the middle of a run, pointing at a directory the user has never heard
 * of. The root here is {@code GPU_BENCH_DATA} if it is set and {@code $HOME/gpu-bench-data}
 * otherwise, which is the same directory on the machine these examples were written on and a
 * sensible one everywhere else.
 *
 * <p>Deliberately not {@code java.io.tmpdir}: these corpora run to tens of gigabytes and {@code
 * /tmp} is a tmpfs on a normal Linux desktop, so a default there would quietly spend the run's
 * memory on its input.
 */
public final class BenchData {

    private BenchData() {}

    /** The root the per-example defaults hang off. */
    public static Path root() {
        final String configured = env("GPU_BENCH_DATA", null);
        if (configured != null) {
            return Paths.get(configured);
        }
        return Paths.get(System.getProperty("user.home", "."), "gpu-bench-data");
    }

    /** A directory under {@link #root()}, named the way the generator names it. */
    public static Path resolve(String first, String... more) {
        return root().resolve(Paths.get(first, more));
    }

    /** An environment variable, treating unset and empty the same way. */
    public static String env(String name, String fallback) {
        final String v = System.getenv(name);
        return v == null || v.isBlank() ? fallback : v;
    }
}
