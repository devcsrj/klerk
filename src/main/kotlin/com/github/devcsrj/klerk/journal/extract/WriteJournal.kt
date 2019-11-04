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

import com.github.devcsrj.klerk.Journal
import org.apache.beam.sdk.transforms.DoFn
import org.apache.beam.sdk.values.KV
import org.slf4j.LoggerFactory
import java.io.File

/**
 * Consolidates all blocks into a single text file
 */
internal class WriteJournal : DoFn<
        KV<Journal, Iterable<@JvmWildcard PageBlock>>,
        File>() {

    private val logger = LoggerFactory.getLogger(WriteJournal::class.java)


    @ProcessElement
    fun processElement(context: ProcessContext) {

        val element = context.element()
        val journal = element.key!!
        val blocks = element.value.sorted()
        val firstPage = blocks.iterator().next()
        val name = "journal-${journal.number}.txt"
        val outputFile = firstPage.file.parentFile.resolve(name)
        outputFile.printWriter().use { writer ->
            for (block in blocks) {
                logger.info("✏️ $block ($journal)")
                writer.println(block.content)
            }
        }
        context.output(outputFile)
    }
}