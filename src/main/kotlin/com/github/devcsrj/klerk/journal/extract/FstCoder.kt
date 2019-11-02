/**
 * Copyright [2019] [Reijhanniel Jearl Campos]
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.devcsrj.klerk.journal.extract

import org.apache.beam.sdk.coders.Coder
import org.apache.beam.vendor.guava.v26_0_jre.com.google.common.io.ByteStreams
import org.apache.beam.vendor.guava.v26_0_jre.com.google.common.primitives.Ints
import org.nustaq.serialization.FSTConfiguration
import java.io.EOFException
import java.io.InputStream
import java.io.OutputStream
import java.nio.ByteBuffer

/**
 * A beam coder using [FSTConfiguration].
 */
internal open class FstCoder<T> : Coder<T>() {

    companion object {

        /**
         * The space allocated for the serialized object
         */
        private const val H_SIZE = 4
        private val FST = FSTConfiguration.createDefaultConfiguration()
    }

    override fun getCoderArguments(): MutableList<out Coder<*>> {
        return mutableListOf()
    }

    override fun verifyDeterministic() {}

    override fun encode(value: T, outStream: OutputStream) {
        val byteArray = FST.asByteArray(value)
        val buffer = ByteBuffer.allocate(H_SIZE)
        buffer.putInt(byteArray.size)
        outStream.write(buffer.array())
        outStream.write(byteArray)
    }

    override fun decode(inStream: InputStream): T {
        var buffer = ByteArray(H_SIZE)
        val read = inStream.read(buffer)
        if (read == -1)
            throw EOFException("Could not read size header for journal")

        val length = Ints.fromByteArray(buffer)
        buffer = ByteArray(length)
        ByteStreams.read(inStream, buffer, 0, length)
        return FST.asObject(buffer) as T
    }

}