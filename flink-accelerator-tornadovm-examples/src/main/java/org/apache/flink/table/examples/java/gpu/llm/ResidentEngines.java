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

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Things that outlive a Flink job but not the TaskManager that ran it.
 *
 * <h2>The problem this solves</h2>
 *
 * <p>A model is 1.4 GiB of weights and a few seconds of device upload. Paying that once per query
 * is not what a deployment does — a TaskManager runs for weeks and answers thousands of queries —
 * so a benchmark that reloads per query is measuring a cold start over and over and calling it
 * inference. Keeping the model resident is the realistic arrangement, and it is also the one the
 * "share the data" instruction asks for.
 *
 * <h2>Why it is spelled like this</h2>
 *
 * <p>A {@code static} field in this class would not survive: Flink disposes the user-code
 * classloader when a job finishes, and the next job gets a fresh copy of every class in the job
 * jar, static fields included. The only maps that outlive that are the ones reachable from a class
 * the <em>parent</em> loader owns, and {@link System#getProperties()} is one — a {@code Hashtable}
 * that happily stores objects, held by a bootstrap class, for the life of the JVM.
 *
 * <p>That is also the constraint on what may go in it. <b>The stored value's class must come from
 * the parent loader too.</b> Storing a value whose class the job jar defines would pin the dead
 * classloader and then fail the next job's cast, because the next job's identically-named class is
 * a different class. So the registry stores only JDK types and types from the engine jars deployed
 * into {@code lib/} — never a type defined in this example jar. The typed accessors are the place
 * that rule is enforced, which is why callers do not touch the map.
 */
public final class ResidentEngines {

    private static final String KEY = "flink.examples.gpu.llm.residentEngines";

    private ResidentEngines() {}

    @SuppressWarnings("unchecked")
    private static Map<String, Object> registry() {
        // computeIfAbsent, not get-then-put: two subtasks of the same operator open concurrently,
        // and the loser of a race must see the winner's map rather than install a second one that
        // the winner's model is not in.
        return (Map<String, Object>)
                System.getProperties()
                        .computeIfAbsent(KEY, k -> new ConcurrentHashMap<String, Object>());
    }

    /** The value stored under {@code name}, or null. */
    public static Object get(String name) {
        return registry().get(name);
    }

    /** Stores {@code value}, which must be of a type the parent classloader defines. */
    public static void put(String name, Object value) {
        registry().put(name, value);
    }

    /** Removes and returns whatever was stored under {@code name}. */
    public static Object remove(String name) {
        return registry().remove(name);
    }

    /** A lock that two subtasks racing to create the same resident object can agree on. */
    public static Object lockFor(String name) {
        return registry().computeIfAbsent(name + ".lock", k -> new Object());
    }
}
