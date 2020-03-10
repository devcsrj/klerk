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

import com.github.devcsrj.docparsr.Json
import com.github.devcsrj.docparsr.ParsingResult
import com.github.devcsrj.klerk.Journal
import com.github.devcsrj.klerk.directoryFor
import org.springframework.batch.item.ItemWriter
import java.nio.file.Files
import java.nio.file.Path

internal class ParsingResultItemWriter(
    private val outputDir: Path
) : ItemWriter<Pair<Journal, ParsingResult>> {

    override fun write(items: MutableList<out Pair<Journal, ParsingResult>>) {
        items.forEach { write(it.first, it.second) }
    }

    private fun write(journal: Journal, item: ParsingResult) {
        val dir = directoryFor(outputDir, journal)
        val file = dir.resolve("${journal.number}.parsed")
        Files.newOutputStream(file).use { sink ->
            item.source(Json).use { src ->
                src.copyTo(sink)
            }
        }
    }
}