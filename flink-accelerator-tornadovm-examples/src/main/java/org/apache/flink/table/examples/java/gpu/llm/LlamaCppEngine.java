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

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * llama.cpp on the same GPU, in the only way a native engine can be reached from a TaskManager.
 *
 * <h2>Why a server and not a one-shot binary</h2>
 *
 * <p>{@code llama-cli} would be the easy comparison and the unfair one: it reloads the model on
 * every invocation, so every query would carry a cold start that the jitllm arm has arranged not to
 * pay. {@code llama-server} is what a deployment would actually run — the model stays resident
 * between requests, exactly as jitllm's does — so that is what this starts, keeps in {@link
 * ResidentEngines}, and reuses across jobs.
 *
 * <p>What remains is a process boundary, and it is not an artefact of this class. llama.cpp is C++;
 * a Flink operator cannot call it in-process without a JNI layer that does not exist. The prompt
 * therefore leaves the JVM as HTTP and the answer comes back the same way. That cost is reported on
 * its own line rather than folded into the engine's, because it belongs to the integration and not
 * to the inference: {@code wall} minus {@code prefill} minus {@code decode} is what the boundary
 * costs.
 *
 * <h2>The server is given every advantage</h2>
 *
 * <p>All layers on the device ({@code -ngl 99}), flash attention on, a prompt batch wide enough
 * that prompt processing is compute-bound, and the same greedy sampling and thinking-off template
 * the other arm uses. If this arm loses it should not be because it was configured to.
 */
public final class LlamaCppEngine implements TriageEngine {

    private static final Pattern CONTENT =
            Pattern.compile("\"content\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"");

    private final Path binary;
    private final Path model;
    private final int contextLength;
    private final int promptBatch;
    private final int port;
    private final String libraryPath;

    private HttpClient http;
    private long loadNanos;
    private boolean reused;

    public LlamaCppEngine(
            Path binary,
            Path model,
            int contextLength,
            int promptBatch,
            int port,
            String libraryPath) {
        this.binary = binary;
        this.model = model;
        this.contextLength = contextLength;
        this.promptBatch = promptBatch;
        this.port = port;
        this.libraryPath = libraryPath;
    }

    private String key() {
        return "llamacpp:" + model.toAbsolutePath() + ":" + contextLength + ":" + port;
    }

    @Override
    public void load() throws Exception {
        long start = System.nanoTime();
        http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
        synchronized (ResidentEngines.lockFor(key())) {
            Process existing = (Process) ResidentEngines.get(key());
            if (existing != null && existing.isAlive() && healthy()) {
                reused = true;
            } else {
                ResidentEngines.remove(key());
                ResidentEngines.put(key(), spawn());
                awaitHealthy();
            }
        }
        loadNanos = System.nanoTime() - start;
    }

    private Process spawn() throws IOException {
        List<String> command = new ArrayList<>();
        command.add(binary.toString());
        command.add("-m");
        command.add(model.toString());
        command.add("--port");
        command.add(Integer.toString(port));
        command.add("--host");
        command.add("127.0.0.1");
        // Every layer on the device. Anything less would be a CPU comparison wearing a GPU label.
        command.add("-ngl");
        command.add("99");
        command.add("-c");
        command.add(Integer.toString(contextLength));
        command.add("-b");
        command.add(Integer.toString(promptBatch));
        command.add("-ub");
        command.add(Integer.toString(promptBatch));
        command.add("-fa");
        command.add("on");
        command.add("--no-webui");
        ProcessBuilder builder = new ProcessBuilder(command);
        if (libraryPath != null && !libraryPath.isEmpty()) {
            // The CUDA 13 toolkit on this machine ships no cuBLAS; the build resolves it from a
            // side-loaded copy, and so must the process.
            builder.environment().merge("LD_LIBRARY_PATH", libraryPath, (a, b) -> b + ":" + a);
        }
        builder.redirectErrorStream(true);
        builder.redirectOutput(new File(System.getProperty("java.io.tmpdir"), "llama-server.log"));
        return builder.start();
    }

    private boolean healthy() {
        try {
            HttpResponse<String> response =
                    http.send(
                            HttpRequest.newBuilder(
                                            URI.create("http://127.0.0.1:" + port + "/health"))
                                    .timeout(Duration.ofSeconds(2))
                                    .GET()
                                    .build(),
                            HttpResponse.BodyHandlers.ofString());
            return response.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    private void awaitHealthy() throws Exception {
        // Loading 1.4 GiB and building the CUDA graphs takes seconds, and the port is open before
        // it is finished, so polling /health is the only honest readiness signal.
        long deadline = System.nanoTime() + Duration.ofSeconds(180).toNanos();
        while (System.nanoTime() < deadline) {
            if (healthy()) {
                return;
            }
            Thread.sleep(50);
        }
        throw new IllegalStateException("llama-server did not become healthy within 180s");
    }

    /** How long {@link #load()} took. Near zero when the server was already running. */
    public long loadNanos() {
        return loadNanos;
    }

    /** Whether the server was already running rather than started by this job. */
    public boolean reusedResidentModel() {
        return reused;
    }

    @Override
    public Completion generate(String prompt, int maxNewTokens) throws Exception {
        // The chat endpoint, so that Qwen3's own template is applied and the two arms see the same
        // token sequence rather than one seeing a template and the other a bare string.
        String body =
                "{\"messages\":[{\"role\":\"user\",\"content\":\""
                        + escape(prompt)
                        + "\"}],\"n_predict\":"
                        + maxNewTokens
                        + ",\"max_tokens\":"
                        + maxNewTokens
                        + ",\"temperature\":0,\"top_k\":1,\"seed\":1,\"stream\":false,"
                        // The server remembers the last prompt's KV and skips the prefill when the
                        // next one shares a prefix. Two runs of this benchmark send the identical
                        // digest, so the second was being charged 9 ms of prefill against the other
                        // arm's 1400 -- a cache hit on a repeat, not a property of the engine. The
                        // other arm resets its context between queries, so this one does too.
                        + "\"cache_prompt\":false,"
                        + "\"chat_template_kwargs\":{\"enable_thinking\":false}}";
        long start = System.nanoTime();
        HttpResponse<String> response =
                http.send(
                        HttpRequest.newBuilder(
                                        URI.create(
                                                "http://127.0.0.1:"
                                                        + port
                                                        + "/v1/chat/completions"))
                                .timeout(Duration.ofMinutes(10))
                                .header("Content-Type", "application/json")
                                .POST(HttpRequest.BodyPublishers.ofString(body))
                                .build(),
                        HttpResponse.BodyHandlers.ofString());
        long wall = System.nanoTime() - start;
        if (response.statusCode() != 200) {
            throw new IllegalStateException(
                    "llama-server returned " + response.statusCode() + ": " + response.body());
        }
        String json = response.body();
        return new Completion(
                unescape(lastContent(json)),
                (int) number(json, "prompt_tokens"),
                (int) number(json, "completion_tokens"),
                (long) (millis(json, "prompt_ms") * 1e6),
                (long) (millis(json, "predicted_ms") * 1e6),
                wall);
    }

    /**
     * The assistant message, which is the last {@code content} in the response.
     *
     * <p>Taking the last match rather than the first, because a response may carry more than one
     * {@code content} -- a reasoning block before the answer, or an echoed message -- and the
     * assistant's text is the last of them. This is the one place a real JSON parser would earn its
     * dependency; the example jars here carry only their own classes, so the shape is pinned here
     * and the assumption written down rather than left to be discovered.
     */
    private static String lastContent(String json) {
        Matcher m = CONTENT.matcher(json);
        String last = null;
        while (m.find()) {
            last = m.group(1);
        }
        if (last == null) {
            throw new IllegalStateException("no content in llama-server response: " + json);
        }
        return last;
    }

    private static double number(String json, String field) {
        Matcher m = Pattern.compile("\"" + field + "\"\\s*:\\s*(-?[0-9.eE+]+)").matcher(json);
        return m.find() ? Double.parseDouble(m.group(1)) : 0.0;
    }

    private static double millis(String json, String field) {
        return number(json, field);
    }

    private static String escape(String s) {
        StringBuilder sb = new StringBuilder(s.length() + 16);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> {
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        return sb.toString();
    }

    private static String unescape(String s) {
        StringBuilder sb = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c != '\\' || i + 1 >= s.length()) {
                sb.append(c);
                continue;
            }
            char next = s.charAt(++i);
            switch (next) {
                case 'n' -> sb.append('\n');
                case 'r' -> sb.append('\r');
                case 't' -> sb.append('\t');
                case 'u' -> {
                    sb.append((char) Integer.parseInt(s.substring(i + 1, i + 5), 16));
                    i += 4;
                }
                default -> sb.append(next);
            }
        }
        return sb.toString();
    }

    @Override
    public String describe() {
        return "llama.cpp llama-server (CUDA, out of process, -ngl 99, batch " + promptBatch + ")";
    }

    /**
     * Drops this job's client and leaves the server running, for the same reason jitllm's stays.
     */
    @Override
    public void close() {
        http = null;
    }

    /** Stops the resident server, for a run that wants a cold start on purpose. */
    public static void evict(Path model, int contextLength, int port) {
        String key = "llamacpp:" + model.toAbsolutePath() + ":" + contextLength + ":" + port;
        synchronized (ResidentEngines.lockFor(key)) {
            Object p = ResidentEngines.remove(key);
            if (p != null) {
                ((Process) p).destroy();
            }
        }
    }
}
