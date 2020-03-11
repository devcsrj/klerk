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

import com.github.devcsrj.klerk.Chamber
import com.github.devcsrj.klerk.Congress
import com.github.devcsrj.klerk.Session
import com.google.common.jimfs.Jimfs
import com.jayway.jsonpath.Configuration
import com.jayway.jsonpath.matchers.JsonPathMatchers.hasJsonPath
import org.hamcrest.CoreMatchers.equalTo
import org.hamcrest.MatcherAssert.assertThat
import org.spekframework.spek2.Spek
import java.net.URI
import java.nio.file.FileSystem
import java.nio.file.Files
import java.time.LocalDate
import java.time.Month
import kotlin.test.assertFalse

object DiskJournalRepositoryTest : Spek({

    val jsonProvider = Configuration.defaultConfiguration().jsonProvider()
    lateinit var fileSystem: FileSystem

    beforeEachTest {
        fileSystem = Jimfs.newFileSystem()
    }

    afterEachTest {
        fileSystem.close()
    }

    test("save") {
        val journal = Journal(
            chamber = Chamber.HOUSE,
            congress = Congress(17),
            session = Session.regular(1),
            number = 1,
            date = LocalDate.of(2016, Month.JULY, 25),
            documentUri = URI.create("https://example.com/journal.pdf")
        )
        val baseDir = fileSystem.getPath("klerk")
        val repo = DiskJournalRepository(baseDir)
        repo.save(journal)

        val expected = baseDir.resolve(fileSystem.getPath("17th", "HOUSE", "regular-1", "journal-001.json"))
        val content = Files.newInputStream(expected).use {
            it.bufferedReader().readText()
        }
        val actual = jsonProvider.parse(content)
        assertThat(actual, hasJsonPath("$.chamber", equalTo("HOUSE")))
        assertThat(actual, hasJsonPath("$.congress.number", equalTo(17)))
        assertThat(actual, hasJsonPath("$.session.number", equalTo(1)))
        assertThat(actual, hasJsonPath("$.session.type", equalTo("REGULAR")))
        assertThat(actual, hasJsonPath("$.number", equalTo(1)))
        assertThat(actual, hasJsonPath("$.date", equalTo("2016-07-25")))
        assertThat(actual, hasJsonPath("$.documentUri", equalTo("https://example.com/journal.pdf")))
    }

    test("iterator") {
        val baseDir = fileSystem.getPath("klerk")
        Files.createDirectory(baseDir)

        val json = baseDir.resolve("journal-001.json")
        Files.newOutputStream(json).bufferedWriter().use {
            it.write(
                """
                {
                  "chamber" : "HOUSE",
                  "congress" : {
                    "number" : 17
                  },
                  "session" : {
                    "number" : 1,
                    "type" : "REGULAR"
                  },
                  "number" : 1,
                  "date" : "2016-07-25",
                  "documentUri" : "https://example.com/journal.pdf"
                }
            """.trimIndent()
            )
        }

        val repo = DiskJournalRepository(baseDir)
        val iterator = repo.iterator()
        val actual = iterator.next()
        val expected = Journal(
            chamber = Chamber.HOUSE,
            congress = Congress(17),
            session = Session.regular(1),
            number = 1,
            date = LocalDate.of(2016, Month.JULY, 25),
            documentUri = URI.create("https://example.com/journal.pdf")
        )
        assertThat(actual, equalTo(expected))
        assertFalse(iterator.hasNext())
    }

    test("assets") {
        val baseDir = fileSystem.getPath("klerk")
        val journalDir = baseDir.resolve(fileSystem.getPath("17th", "HOUSE", "regular-1"))
        Files.createDirectories(journalDir)

        val journal = Journal(
            chamber = Chamber.HOUSE,
            congress = Congress(17),
            session = Session.regular(1),
            number = 1,
            date = LocalDate.of(2016, Month.JULY, 25),
            documentUri = URI.create("https://example.com/journal.pdf")
        )
        val repo = DiskJournalRepository(baseDir)
        val assets = repo.assets(journal)
        assets.sink("test.txt").use {
            it.write("hello".toByteArray())
        }

        val expected = journalDir.resolve("journal-001__test.txt")
        val actual = String(Files.readAllBytes(expected))
        assertThat(actual, equalTo("hello"))
    }
})