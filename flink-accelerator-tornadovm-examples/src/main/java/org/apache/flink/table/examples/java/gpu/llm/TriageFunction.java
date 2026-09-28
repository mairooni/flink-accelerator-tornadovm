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

import org.apache.flink.table.functions.FunctionContext;
import org.apache.flink.table.functions.ScalarFunction;

import java.nio.file.Path;
import java.util.Arrays;

/**
 * The step that turns a digest of anomalies into a maintenance note, inside the job.
 *
 * <h2>Why the inference is a UDF and not something the client does after {@code collect()}</h2>
 *
 * <p>Running it in the client would put the model in a different JVM from the operator that scored
 * the telemetry, which throws away the only thing this comparison is about. As a UDF it runs on the
 * TaskManager — the same JVM, the same TornadoVM runtime, the same CUDA context as the offloaded
 * {@code Calc} — and the digest reaches it as a Java string that was never serialised, never
 * written to disk and never sent over a socket.
 *
 * <p>It also keeps the query honest. Nothing here is a GPU hint: {@code triage(...)} is an ordinary
 * scalar function over an ordinary {@code LISTAGG}, and the accelerator decides what to offload
 * from the shape of the plan beneath it, with no help from the text of the query.
 *
 * <h2>The prompt is deterministic even though the aggregation is not</h2>
 *
 * <p>{@code LISTAGG} concatenates in whatever order the rows arrive, which is not fixed. Two runs
 * would then send two different prompts and their token counts would not be comparable. So the
 * lines are split, sorted and rejoined here: the digest's <em>content</em> is what the aggregation
 * determines, and its <em>order</em> is determined by this function, which makes the prompt a
 * function of the data alone.
 */
public final class TriageFunction extends ScalarFunction {

    private static final long serialVersionUID = 1L;

    /** The separator {@code LISTAGG} joins digest lines with. Chosen not to occur in a line. */
    public static final String SEPARATOR = " ;; ";

    private final String engineKind;
    private final String modelPath;
    private final String binaryPath;
    private final String libraryPath;
    private final int contextLength;
    private final int promptBatch;
    private final int maxNewTokens;
    private final int port;
    private final boolean deviceSampling;
    private final boolean cold;
    private final boolean nativeLibraries;

    private transient TriageEngine engine;
    private transient long loadNanos;
    private transient boolean reused;

    public TriageFunction(
            String engineKind,
            String modelPath,
            String binaryPath,
            String libraryPath,
            int contextLength,
            int promptBatch,
            int maxNewTokens,
            int port,
            boolean deviceSampling,
            boolean cold,
            boolean nativeLibraries) {
        this.engineKind = engineKind;
        this.modelPath = modelPath;
        this.binaryPath = binaryPath;
        this.libraryPath = libraryPath;
        this.contextLength = contextLength;
        this.promptBatch = promptBatch;
        this.maxNewTokens = maxNewTokens;
        this.port = port;
        this.deviceSampling = deviceSampling;
        this.cold = cold;
        this.nativeLibraries = nativeLibraries;
    }

    @Override
    public void open(FunctionContext context) throws Exception {
        if (cold) {
            // A cold run on purpose: drop whatever an earlier job left resident, so that the load
            // column measures a first query on a fresh TaskManager rather than a second one.
            if ("jitllm".equals(engineKind)) {
                JitllmEngine.evict(Path.of(modelPath), contextLength, promptBatch, nativeLibraries);
            } else {
                LlamaCppEngine.evict(Path.of(modelPath), contextLength, port);
            }
        }
        engine =
                "jitllm".equals(engineKind)
                        ? new JitllmEngine(
                                Path.of(modelPath),
                                contextLength,
                                promptBatch,
                                deviceSampling,
                                nativeLibraries)
                        : new LlamaCppEngine(
                                Path.of(binaryPath),
                                Path.of(modelPath),
                                contextLength,
                                promptBatch,
                                port,
                                libraryPath);
        engine.load();
        loadNanos =
                engine instanceof JitllmEngine
                        ? ((JitllmEngine) engine).loadNanos()
                        : ((LlamaCppEngine) engine).loadNanos();
        reused =
                engine instanceof JitllmEngine
                        ? ((JitllmEngine) engine).reusedResidentModel()
                        : ((LlamaCppEngine) engine).reusedResidentModel();
    }

    /** One digest in, one triage note out, with the engine's accounting appended. */
    public String eval(String digest) {
        if (digest == null || digest.isEmpty()) {
            return "no anomalies";
        }
        String[] lines = digest.split(java.util.regex.Pattern.quote(SEPARATOR));
        Arrays.sort(lines);
        String prompt = TelemetryTriage.prompt(lines);
        try {
            TriageEngine.Completion completion = engine.generate(prompt, maxNewTokens);
            return TelemetryTriage.report(
                            engine.describe(), reused, loadNanos, prompt.length(), completion)
                    + completion.text;
        } catch (Exception e) {
            throw new RuntimeException("inference failed", e);
        }
    }

    @Override
    public void close() {
        if (engine != null) {
            engine.close();
            engine = null;
        }
    }
}
