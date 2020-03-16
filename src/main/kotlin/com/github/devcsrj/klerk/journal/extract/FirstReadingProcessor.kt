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
package com.github.devcsrj.klerk.journal.extract

import com.github.devcsrj.docparsr.Document
import com.github.devcsrj.klerk.bill.BillEvent
import com.github.devcsrj.klerk.journal.Journal
import com.github.devcsrj.klerk.journal.JournalAssets
import com.github.devcsrj.klerk.journal.JournalRepository
import org.slf4j.LoggerFactory
import org.springframework.batch.item.ItemProcessor
import java.nio.file.Files

internal class FirstReadingProcessor(
    private val repository: JournalRepository
) : ItemProcessor<Journal, List<BillEvent>> {

    companion object {

        @JvmStatic
        private val labels = setOf(
            KlerkParsr.FirstReading.BILL_LINE_LABEL,
            KlerkParsr.FirstReading.HEADING_LABEL,
            KlerkParsr.FirstReading.INTRODUCER_LABEL,
            KlerkParsr.FirstReading.RECEIVING_COMMITTEE_LABEL
        )

        @JvmStatic
        private val logger = LoggerFactory.getLogger(FirstReadingProcessor::class.java)
    }

    override fun process(item: Journal): List<BillEvent> {
        val assets = repository.assets(item)
        val file = assets.file(JournalAssets.PARSING_RESULT)
        if (!Files.exists(file)) {
            return emptyList()
        }

        logger.info("Collecting bills on first reading from '$item'...")
        val document = Document.from(file)
        return emptyList()
    }

}