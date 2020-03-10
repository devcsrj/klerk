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

import org.springframework.batch.item.ItemReader
import java.io.File
import java.nio.file.Files
import java.nio.file.Path
import kotlin.streams.toList

internal class DirectoryPdfItemReader(
    private val baseDir: Path
) : ItemReader<File> {

    private val iterator: Iterator<Path> = Files.walk(baseDir)
        .filter { it.toString().endsWith(".pdf") }
        .toList()
        .iterator()

    override fun read(): File? {
        return if (iterator.hasNext()) iterator.next().toFile() else null
    }
}