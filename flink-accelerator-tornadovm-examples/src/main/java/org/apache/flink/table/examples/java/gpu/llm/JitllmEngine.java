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

import org.beehive.jitllm.api.GenerationRequest;
import org.beehive.jitllm.api.GenerationResult;
import org.beehive.jitllm.api.GenerationSession;
import org.beehive.jitllm.api.LocalModel;
import org.beehive.jitllm.api.LocalModels;
import org.beehive.jitllm.api.ModelOptions;
import org.beehive.jitllm.api.TextGenerationModel;
import org.beehive.jitllm.api.ThinkingMode;
import org.beehive.jitllm.runtime.backend.BackendId;

import java.nio.file.Path;

/**
 * jitllm, loaded into the TaskManager's own JVM.
 *
 * <h2>Why in-process is the point and not a shortcut</h2>
 *
 * <p>The offloaded {@code Calc} that scored the telemetry ran on this GPU, through this JVM's
 * TornadoVM runtime, in this CUDA context. Loading the model here means the answer to "where is the
 * data" is: it never left. There is no socket, no JSON, no second copy of the digest, no second
 * CUDA context competing for the same 8 GiB, and no second process whose page cache has to be
 * warmed. A native engine cannot do this at any price — it is not a Java library — and that is the
 * structural advantage this arm is built to show.
 *
 * <p>The practical consequences are all in the four things this class does:
 *
 * <ol>
 *   <li><b>The batched-prefill MMA path.</b> {@code jitllm.withPrefillDecode} plus {@code
 *       jitllm.prefillBatchSize} moves prompt processing from one token per forward to {@code B} of
 *       them, which on this card is the difference between 138 and 5562 prompt tokens per second.
 *       Left at the default, a digest of a few hundred tokens would spend seconds in prefill and
 *       the arm would lose on that alone. These are read <em>at class initialisation</em> of the
 *       state buffers, so they are set before any jitllm type is touched.
 *   <li><b>One resident model per JVM.</b> {@link ResidentEngines} keeps the loaded model alive
 *       across Flink jobs, so the second query on a cluster pays nothing for the model at all. This
 *       is what a long-lived TaskManager actually looks like; reloading 1.4 GiB per query would be
 *       an artefact of the benchmark, not of the deployment.
 *   <li><b>A context sized to the work.</b> The KV cache is the second-largest allocation after the
 *       weights, and it is sized by the context length, not by the prompt. Asking for the model's
 *       own maximum would reserve device memory that the offloaded operator also wants.
 *   <li><b>Greedy sampling.</b> Temperature zero and a fixed seed, so that two runs of this arm
 *       produce the same text and a difference between runs is a defect rather than a sample.
 * </ol>
 */
public final class JitllmEngine implements TriageEngine {

    private final Path model;
    private final int contextLength;
    private final int prefillBatch;
    private final boolean deviceSampling;

    private LocalModel loaded;
    private GenerationSession session;
    private long loadNanos;
    private boolean reused;

    public JitllmEngine(Path model, int contextLength, int prefillBatch, boolean deviceSampling) {
        this.model = model;
        this.contextLength = contextLength;
        this.prefillBatch = prefillBatch;
        this.deviceSampling = deviceSampling;
    }

    /**
     * Sets the properties the batched path is gated on, before anything reads them.
     *
     * <p>They are system properties rather than builder arguments because the buffers they size are
     * allocated in a static initialiser; jitllm's own benchmark harness sets them in the same way
     * and for the same reason. Setting them here is safe only because no jitllm class has been
     * initialised yet at this point — this class merely references those types, which loads them
     * without running their initialisers. The deployment also sets them on the TaskManager command
     * line, which is belt and braces for exactly this ordering hazard.
     */
    private void arm() {
        if (prefillBatch > 1) {
            System.setProperty("jitllm.withPrefillDecode", "true");
            System.setProperty("jitllm.prefillBatchSize", Integer.toString(prefillBatch));
        }
        if (deviceSampling) {
            System.setProperty("jitllm.deviceSample", "true");
        }
    }

    @Override
    public void load() throws Exception {
        arm();
        final String key =
                "jitllm:" + model.toAbsolutePath() + ":" + contextLength + ":" + prefillBatch;
        long start = System.nanoTime();
        // Two subtasks opening at once must not both load 1.4 GiB onto an 8 GiB card. One wins the
        // lock and loads; the other waits and finds the model already there.
        synchronized (ResidentEngines.lockFor(key)) {
            Object existing = ResidentEngines.get(key);
            if (existing != null) {
                loaded = (LocalModel) existing;
                session = (GenerationSession) ResidentEngines.get(key + ".session");
                reused = true;
            } else {
                loaded =
                        LocalModels.load(
                                model,
                                ModelOptions.builder()
                                        .backend(BackendId.CUDA)
                                        .contextLength(contextLength)
                                        .thinkingMode(ThinkingMode.DISABLED)
                                        .build());
                session = ((TextGenerationModel) loaded).newSession();
                // Compile the execution plan now rather than inside the first generate().
                //
                // The kernels a transformer needs are JIT-compiled by TornadoVM on first use, and
                // that is seconds of work. Left to happen inside generate(), it lands in the
                // inference wall time and in none of the phases the engine reports -- the first
                // measurement here had 13.8 s of inference wall against 3.2 s of prefill and
                // decode, and the missing ten seconds were this. Doing it in load() puts the cost
                // in the column that says "load", where a reader can see it is paid once.
                session.prepare();
                ResidentEngines.put(key, loaded);
                ResidentEngines.put(key + ".session", session);
            }
        }
        loadNanos = System.nanoTime() - start;
    }

    /**
     * How long {@link #load()} took. Near zero when the model was already resident from an earlier
     * job on this TaskManager, which is the case the {@code warm} column of the report names.
     */
    public long loadNanos() {
        return loadNanos;
    }

    /** Whether the model was already resident rather than loaded by this job. */
    public boolean reusedResidentModel() {
        return reused;
    }

    @Override
    public Completion generate(String prompt, int maxNewTokens) {
        // A session continues its own sequence, and this one outlives the job that created it.
        // Without the reset the second query arrives with the first query's conversation still in
        // the context: the prompt is silently longer, the prefill is three times slower, and by
        // the third query the context is full and the model generates nothing at all. All three
        // were observed before this line existed. A triage query is stateless, so the sequence is
        // discarded and every run starts from an empty context.
        session.reset();
        long start = System.nanoTime();
        GenerationResult result =
                session.generate(
                        GenerationRequest.builder()
                                .prompt(prompt)
                                .maxNewTokens(maxNewTokens)
                                .temperature(0.0f)
                                .seed(1L)
                                .build());
        long wall = System.nanoTime() - start;
        return new Completion(
                result.text(),
                result.promptTokens(),
                result.generatedTokens(),
                result.timings().prefill().toNanos(),
                result.timings().decode().toNanos(),
                wall);
    }

    @Override
    public String describe() {
        return "jitllm (TornadoVM CUDA, in-process, prefill batch "
                + prefillBatch
                + (deviceSampling ? ", device sampling" : "")
                + ")";
    }

    /**
     * Drops this job's handles, and deliberately does not close the model.
     *
     * <p>Residency is the whole point: the model and its session belong to the JVM, not to the job
     * that happened to load them, and closing them here would give the next query a cold start
     * again. They are released when the TaskManager exits. {@link #evict} is the way to reclaim
     * them on purpose.
     */
    @Override
    public void close() {
        session = null;
        loaded = null;
    }

    /** Releases the resident model, for a run that wants a cold start on purpose. */
    public static void evict(Path model, int contextLength, int prefillBatch) {
        final String key =
                "jitllm:" + model.toAbsolutePath() + ":" + contextLength + ":" + prefillBatch;
        synchronized (ResidentEngines.lockFor(key)) {
            // A session is closed before its model: the session holds a lease on the KV pool that
            // the model owns, and releasing them the other way round leaks the lease.
            Object s = ResidentEngines.remove(key + ".session");
            if (s != null) {
                ((GenerationSession) s).close();
            }
            Object m = ResidentEngines.remove(key);
            if (m != null) {
                ((LocalModel) m).close();
            }
        }
    }
}
