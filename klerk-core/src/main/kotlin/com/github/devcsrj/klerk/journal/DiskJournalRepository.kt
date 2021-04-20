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
package com.github.devcsrj.klerk.journal

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule
import com.fasterxml.jackson.module.kotlin.KotlinModule
import com.github.devcsrj.klerk.ordinal
import org.slf4j.LoggerFactory
import java.io.IOException
import java.io.OutputStream
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import java.util.*
import java.util.function.Predicate
import java.util.regex.Pattern
import kotlin.streams.toList

/**
 * A [JournalRepository] implementation that relies on the directory tree to
 * store [Journal]s.
 */
internal class DiskJournalRepository(
    private val baseDir: Path
) : JournalRepository {

    companion object {

        private const val INFO_EXT = "json"
        private val NAME_PATTERN = Pattern.compile("journal-[\\d]{3}\\.json")
    }

    private val logger = LoggerFactory.getLogger(DiskJournalRepository::class.java)
    private val mapper: ObjectMapper = ObjectMapper().apply {
        registerModule(KotlinModule())
        registerModule(JavaTimeModule())
    }

    override fun save(journal: Journal) {
        val dir = directoryFor(journal)
        Files.createDirectories(dir)

        val json = dir.resolve("${simpleNameOf(journal)}.$INFO_EXT")
        Files.newOutputStream(json).use { source ->
            mapper.writerWithDefaultPrettyPrinter().writeValue(source, journal)
        }
    }

    override fun assets(journal: Journal): Assets {
        return DiskAssets(journal)
    }

    override fun iterator(): Iterator<Journal> {
        return sequence<Journal> {
            val stack = Stack<Path>()
            stack.add(baseDir)

            while (stack.isNotEmpty()) {
                val next = stack.pop()
                if (Files.isDirectory(next)) {
                    stack.addAll(Files.list(next).toList())
                } else {
                    val matcher = NAME_PATTERN.matcher(next.fileName.toString())
                    if (matcher.find()) {
                        try {
                            Files.newInputStream(next).use {
                                yield(mapper.readValue(it, Journal::class.java))
                            }
                        } catch (e: IOException) {
                            logger.warn(
                                "Skipping unreadable journal '{}' (cause: {})",
                                baseDir.relativize(next), e.message
                            )
                        }
                    }
                }
            }
        }.iterator()
    }

    /**
     * Constructs the directory structure for [Journal]
     */
    private fun directoryFor(journal: Journal): Path {
        val congress = journal.congress.number.ordinal()
        val session = journal.session.let {
            "${it.type.name.toLowerCase()}-${it.number}"
        }
        val chamber = journal.chamber.name
        return baseDir
            .resolve(congress)
            .resolve(chamber)
            .resolve(session)
    }

    private fun simpleNameOf(journal: Journal): String {
        return "journal-" + "${journal.number}".padStart(3, '0')
    }

    private inner class DiskAssets(private val journal: Journal) :
        Assets {

        override fun sink(name: String): OutputStream {
            return Files.newOutputStream(
                file(name),
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING
            )
        }

        override fun file(name: String): Path {
            return directoryFor(journal).resolve("${prefix()}$name")
        }

        override fun list(filenameFilter: Predicate<String>): List<Path> {
            return Files.list(directoryFor(journal))
                .filter(this::ownsFile)
                .filter { matchesFilter(it, filenameFilter) }
                .toList()
        }

        private fun ownsFile(file: Path) = file.fileName.toString().startsWith(prefix())

        private fun matchesFilter(file: Path, filenameFilter: Predicate<String>): Boolean {
            val suffix = file.fileName.toString().substringAfter(prefix())
            return filenameFilter.test(suffix)
        }

        private fun prefix() = "${simpleNameOf(journal)}__"
    }
}
