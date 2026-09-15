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

package org.apache.flink.table.gpu.codegen;

import org.apache.flink.annotation.Internal;

/**
 * The width a staged column is held at on the device.
 *
 * <p>This exists because "everything is a double" was not a simplification, it was a defect. The
 * generator admits {@code INTEGER}, {@code FLOAT} and {@code REAL} columns on the grounds that each
 * is exact in a double, which is true of the <em>values</em> and says nothing about how they are
 * <em>fetched</em>: a gather reading a four-byte field with {@code getDouble} reads eight bytes and
 * returns whatever follows it. An {@code INT} column holding 42 arrived on the device as 2.08e-322.
 *
 * <p>Carrying the declared type instead fixes that, and pays twice over. A {@code FLOAT} or {@code
 * INT} column now transfers four bytes per row rather than eight, which matters because every
 * workload measured so far is bound by moving its input rather than by arithmetic.
 *
 * <p>The <em>arithmetic</em> is a separate question and is deliberately not changed here. Values
 * are widened to {@code double} on entry to the kernel, so the expression is evaluated exactly as
 * it was before; only storage and transfer follow the declared type.
 */
@Internal
public enum GpuValueType {
    INT("IntArray", "int"),
    FLOAT("FloatArray", "float"),
    DOUBLE("DoubleArray", "double");

    private final String arrayType;
    private final String primitive;

    GpuValueType(String arrayType, String primitive) {
        this.arrayType = arrayType;
        this.primitive = primitive;
    }

    /** Simple name of the TornadoVM array class holding a column of this type. */
    public String arrayType() {
        return arrayType;
    }

    /** The Java primitive a single element of that array reads as. */
    public String primitive() {
        return primitive;
    }
}
