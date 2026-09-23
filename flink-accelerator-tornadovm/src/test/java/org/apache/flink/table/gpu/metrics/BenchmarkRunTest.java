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

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * The run record has to be complete and the statistics have to be the stated ones.
 *
 * <p>A metadata collector that quietly answers "unavailable" for every field is worse than no
 * collector at all: it produces a header that looks like provenance and carries none. So the keys
 * are asserted even where their values cannot be on a machine without a card — "gpu: unavailable"
 * is a fact about the run and must appear, rather than the line being dropped.
 */
class BenchmarkRunTest {

    @Test
    @DisplayName("the environment record names every field a reader would have to ask for")
    void environmentIsComplete() {
        String environment = BenchmarkRun.environment();
        System.out.println(environment);

        assertThat(environment)
                .contains("host:")
                .contains("cpu:")
                .contains("load:")
                .contains("gpu:")
                .contains("power/clocks:")
                .contains("cuda:")
                .contains("rapids/cudf:")
                .contains("jvm:")
                .contains("jvm flags:")
                .contains("heap max:")
                .contains("provider:")
                .contains("tornadovm:")
                .contains("git:");
    }

    @Test
    @DisplayName("a dataset is identified by its contents, not by its row count")
    void datasetChecksumFollowsTheContents(@TempDir Path dir) throws Exception {
        Files.write(dir.resolve("part-0.csv"), "1,2.0\n3,4.0\n".getBytes("UTF-8"));
        String first = BenchmarkRun.dataset(dir.toString());
        assertThat(first).contains("1 files").contains("crc32=");

        // The same shape and a different value must not produce the same checksum, or two runs
        // that read different inputs would claim to have read the same one.
        Files.write(dir.resolve("part-0.csv"), "1,2.0\n3,4.1\n".getBytes("UTF-8"));
        String second = BenchmarkRun.dataset(dir.toString());
        assertThat(crc(second)).isNotEqualTo(crc(first));

        // And it must be stable, or it identifies nothing.
        assertThat(crc(BenchmarkRun.dataset(dir.toString()))).isEqualTo(crc(second));
    }

    @Test
    @DisplayName("an absent dataset says so rather than being omitted")
    void anAbsentDatasetIsReported() {
        assertThat(BenchmarkRun.dataset("/nonexistent/path")).contains("(absent)");
    }

    @Test
    @DisplayName("the distribution excludes warm-up and reports the cold run separately")
    void statisticsAreTheStatedOnes() {
        BenchmarkRun run = new BenchmarkRun("test", 1);
        // A cold run an order of magnitude above the rest, then 10, 20, 30, 40, 50 ms.
        long[] millis = {500, 10, 20, 30, 40, 50};
        for (long ms : millis) {
            run.record(ms * 1_000_000L);
        }
        String summary = run.summary();
        System.out.println(summary);
        // Column widths are formatting; the numbers are the contract.
        String flat = summary.replaceAll("[ \\t]+", " ");

        assertThat(flat).contains("runs=6").contains("warm-up=1").contains("measured=5");
        // Reported, not discarded: a cold start fifty times a warm one is a deployment fact.
        assertThat(flat).contains("first (cold) 500.0 ms");
        // Nearest-rank over five values: the third is the median, the fifth is p95.
        assertThat(flat).contains("median 30.0 ms");
        assertThat(flat).contains("p95 50.0 ms");
        assertThat(flat).contains("min 10.0 ms");
        assertThat(flat).contains("max 50.0 ms");
        assertThat(flat).contains("mean 30.0 ms");
        assertThat(flat).contains("stddev 14.1 ms");
    }

    @Test
    @DisplayName("a run with nothing recorded says so instead of dividing by zero")
    void noRunsIsNotAnError() {
        assertThat(new BenchmarkRun("test", 1).summary()).contains("no runs recorded");
    }

    private static String crc(String record) {
        int at = record.indexOf("crc32=");
        return at < 0 ? record : record.substring(at, at + 14);
    }
}
