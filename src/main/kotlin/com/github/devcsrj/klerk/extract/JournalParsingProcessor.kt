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
package com.github.devcsrj.klerk.extract

import com.github.devcsrj.docparsr.DocParsr
import com.github.devcsrj.docparsr.ParsingResult
import com.github.devcsrj.klerk.Journal
import com.github.devcsrj.klerk.fromJson
import org.slf4j.LoggerFactory
import org.springframework.batch.item.ItemProcessor
import java.io.File

internal class JournalParsingProcessor(
    private val parser: DocParsr
) : ItemProcessor<File, Pair<Journal, ParsingResult>> {

    private val logger = LoggerFactory.getLogger(JournalParsingProcessor::class.java)

    override fun process(item: File): Pair<Journal, ParsingResult>? {
        val name = item.nameWithoutExtension
        val journal = Journal.fromJson(item.parentFile.resolve("${name}.json").readText())
        logger.info("Parsing '$journal'...")

        val result = parser.newParsingJob(item, KlerkParsr.CONFIG).execute()
        return Pair(journal, result)
    }
}