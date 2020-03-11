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
import com.github.devcsrj.klerk.Assets
import com.github.devcsrj.klerk.Journal
import com.github.devcsrj.klerk.JournalRepository
import com.github.devcsrj.klerk.KlerkAssets
import org.slf4j.LoggerFactory
import org.springframework.batch.item.ItemProcessor
import java.nio.file.Files

internal class JournalParsingProcessor(
    private val parser: DocParsr,
    private val repository: JournalRepository
) : ItemProcessor<Journal, Pair<Assets, ParsingResult>> {

    private val logger = LoggerFactory.getLogger(JournalParsingProcessor::class.java)

    override fun process(item: Journal): Pair<Assets, ParsingResult>? {
        val assets = repository.assets(item)
        val file = assets.file(KlerkAssets.DOCUMENT)
        if (!Files.exists(file)) {
            return null
        }

        logger.info("Parsing '$item'...")
        val result = parser.newParsingJob(file.toFile(), KlerkParsr.CONFIG).execute()
        return Pair(assets, result)
    }
}