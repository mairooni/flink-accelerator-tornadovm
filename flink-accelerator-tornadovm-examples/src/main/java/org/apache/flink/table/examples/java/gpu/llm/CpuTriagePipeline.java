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
 * Arm B: score the telemetry on the CPU, then write the note with llama.cpp on the GPU.
 *
 * <p>This is the arrangement the comparison exists to beat, and it is the arrangement most people
 * have: Flink does its work on the cores it was given, and the model lives in a separate native
 * process that the job talks to. The SQL is character-for-character the same as {@link
 * GpuTriagePipeline}'s — the accelerator is simply not enabled, so the planner builds the
 * code-generated operator and the scoring runs where it always did.
 *
 * <p>llama.cpp is given every advantage available to it: all layers on the device, flash attention,
 * a resident server rather than a per-query process, and the same greedy sampling. What it cannot
 * be given is a way into the JVM.
 *
 * <pre>
 *   flink run CpuTriagePipeline.jar --modes 8 --threshold 9.0
 * </pre>
 */
public final class CpuTriagePipeline {

    public static void main(String[] args) throws Exception {
        final TelemetryTriage.Args parsed = TelemetryTriage.Args.parse(args);
        if (parsed.generate) {
            TelemetryTriage.generate(parsed);
            return;
        }
        System.out.print(parsed.banner("cpu-preprocess + llama.cpp"));
        TelemetryTriage.run(
                parsed,
                false,
                new TriageFunction(
                        "llamacpp",
                        parsed.model,
                        parsed.binary,
                        parsed.libraryPath,
                        parsed.contextLength,
                        parsed.promptBatch,
                        parsed.maxNewTokens,
                        parsed.port,
                        parsed.deviceSampling,
                        parsed.cold,
                        parsed.nativeLibraries));
    }
}
