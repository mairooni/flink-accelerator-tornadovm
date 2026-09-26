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
 * Arm A: score the telemetry on the GPU, then write the note on the same GPU, in the same JVM.
 *
 * <p>Both halves of the pipeline run on the device, and neither of them leaves the TaskManager.
 * The offloaded {@code Calc} produces the anomalies; TornadoVM's runtime is already initialised
 * and its CUDA context already current when jitllm loads the model into it; the digest reaches the
 * model as a Java string. There is no second process, no socket and no second copy of anything.
 *
 * <p>The comparison this is half of is {@link CpuTriagePipeline}, which runs the identical SQL
 * without the accelerator and reaches llama.cpp over HTTP. The two differ in the two things being
 * measured and in nothing else.
 *
 * <pre>
 *   flink run GpuTriagePipeline.jar --generate --rows 8000000
 *   flink run GpuTriagePipeline.jar --modes 8 --threshold 9.0
 * </pre>
 */
public final class GpuTriagePipeline {

    public static void main(String[] args) throws Exception {
        final TelemetryTriage.Args parsed = TelemetryTriage.Args.parse(args);
        if (parsed.generate) {
            TelemetryTriage.generate(parsed);
            return;
        }
        System.out.print(parsed.banner("gpu-preprocess + jitllm"));
        TelemetryTriage.run(
                parsed,
                true,
                new TriageFunction(
                        "jitllm",
                        parsed.model,
                        parsed.binary,
                        parsed.libraryPath,
                        parsed.contextLength,
                        parsed.promptBatch,
                        parsed.maxNewTokens,
                        parsed.port,
                        parsed.deviceSampling,
                        parsed.cold));
    }
}
