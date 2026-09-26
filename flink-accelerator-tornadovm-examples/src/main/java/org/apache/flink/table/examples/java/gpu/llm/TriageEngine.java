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

package org.apache.flink.table.examples.java.gpu.llm;

/**
 * One LLM, behind the smallest interface that both arms of the comparison can implement.
 *
 * <p>The two engines differ in where they run, not in what they are asked. jitllm is a Java library
 * and loads into the TaskManager's own JVM, beside the TornadoVM runtime that the offloaded
 * operator is already using. llama.cpp is a native binary and cannot be anything but a second
 * process. That asymmetry is the measurement, so it must not be smuggled into the interface: both
 * sides see {@code load}, {@code generate}, {@code close}, and the cost of whatever each has to do
 * to honour them lands in the timings.
 *
 * <h2>Why the timings are a record and not a log line</h2>
 *
 * <p>The end-to-end number this benchmark reports is the wall time of {@code flink run}, and
 * instrumenting the phases inside that run would perturb it. So the phases are recorded as plain
 * fields, accumulated without formatting or I/O, and printed once after the job has finished.
 */
public interface TriageEngine extends AutoCloseable {

    /**
     * Brings the model to the point where a prompt can be answered.
     *
     * <p>Called on a thread that does not hold up the query, so that the cost of loading overlaps
     * the scan. What "loaded" means differs per engine — a GGUF parse and a device upload for
     * jitllm, a process launch and a health poll for llama.cpp — and the difference is exactly what
     * the {@code load} column of the breakdown is for.
     */
    void load() throws Exception;

    /** One prompt in, one completion out, with whatever the engine can say about the cost. */
    Completion generate(String prompt, int maxNewTokens) throws Exception;

    /** A short name for the report. */
    String describe();

    @Override
    void close();

    /**
     * What came back, and what it cost.
     *
     * <p>{@code prefillNanos} and {@code decodeNanos} are the engine's own accounting and do not
     * have to add up to {@code wallNanos}: the difference is tokenisation, template rendering and,
     * for llama.cpp, the HTTP round trip. Reporting all three separately is what lets a gap be
     * attributed rather than guessed at.
     */
    final class Completion {
        public final String text;
        public final int promptTokens;
        public final int generatedTokens;
        public final long prefillNanos;
        public final long decodeNanos;
        public final long wallNanos;

        public Completion(
                String text,
                int promptTokens,
                int generatedTokens,
                long prefillNanos,
                long decodeNanos,
                long wallNanos) {
            this.text = text;
            this.promptTokens = promptTokens;
            this.generatedTokens = generatedTokens;
            this.prefillNanos = prefillNanos;
            this.decodeNanos = decodeNanos;
            this.wallNanos = wallNanos;
        }
    }
}
