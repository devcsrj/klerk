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

import com.github.devcsrj.docparsr.Text
import com.github.devcsrj.docparsr.ParsingResult
import com.github.devcsrj.klerk.journal.Assets
import com.github.devcsrj.klerk.journal.JournalAssets
import org.springframework.batch.item.ItemWriter

internal class ParsingResultItemWriter : ItemWriter<Pair<Assets, ParsingResult>> {

    override fun write(items: MutableList<out Pair<Assets, ParsingResult>>) {
        items.forEach { write(it.first, it.second) }
    }

    private fun write(assets: Assets, item: ParsingResult) {
        assets.sink(JournalAssets.PARSED_TEXT).use { sink ->
            item.source(Text).use { src ->
                src.copyTo(sink)
            }
        }
    }
}