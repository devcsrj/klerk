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
package com.github.devcsrj.klerk.journal.preprocess

import com.github.devcsrj.klerk.journal.Assets
import com.github.devcsrj.klerk.journal.JournalAssets
import com.github.devcsrj.klerk.journal.JournalRepository
import org.springframework.batch.item.ItemReader
import java.nio.file.Path

internal class JournalPageImageFileReader(
    private val journalRepo: JournalRepository
) : ItemReader<Pair<Assets, Path>> {

    private val journalIterator = journalRepo.iterator()

    private var assets = Assets.EMPTY
    private var fileIterator: Iterator<Path> = arrayOf<Path>().iterator()

    override fun read(): Pair<Assets, Path>? {
        if (fileIterator.hasNext()) {
            return Pair(assets, fileIterator.next())
        }

        if (!journalIterator.hasNext()) {
            return null
        }

        val next = journalIterator.next()
        assets = journalRepo.assets(next)
        fileIterator = assets.list(JournalAssets.documentPagePngFilter()).iterator()
        return read()
    }
}