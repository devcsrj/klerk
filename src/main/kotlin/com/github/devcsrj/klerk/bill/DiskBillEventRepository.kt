/**
 * Copyright [2020] [Reijhanniel Jearl Campos]
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
package com.github.devcsrj.klerk.bill

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule
import com.fasterxml.jackson.module.kotlin.KotlinModule
import org.springframework.beans.factory.DisposableBean
import org.springframework.beans.factory.InitializingBean
import java.nio.ByteBuffer
import java.nio.channels.ByteChannel
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption

class DiskBillEventRepository(
    private val baseDir: Path
) : BillEventRepository, InitializingBean, DisposableBean {

    private val mapper: ObjectMapper = ObjectMapper().apply {
        registerModule(KotlinModule())
        registerModule(JavaTimeModule())
    }
    private lateinit var fileChannel: ByteChannel
    private val lock = this // guard

    override fun afterPropertiesSet() {
        val file = baseDir.resolve("bill-events.jsonl")
        val options = arrayOf(StandardOpenOption.CREATE, StandardOpenOption.APPEND)
        fileChannel = Files.newByteChannel(file, *options)
    }

    override fun destroy() {
        fileChannel.close()
    }

    override fun save(event: BillEvent) {
        val bytes = mapper.writeValueAsBytes(event) + System.lineSeparator().toByteArray()
        synchronized(lock) {
            val buffer = ByteBuffer.wrap(bytes)
            fileChannel.write(buffer)
        }
    }
}