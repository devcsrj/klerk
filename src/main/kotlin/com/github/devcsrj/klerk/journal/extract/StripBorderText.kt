/**
 * Klerk
 * Copyright (C) 2019 Reijhanniel Jearl Campos
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package com.github.devcsrj.klerk.journal.extract

import com.fasterxml.jackson.databind.ObjectMapper
import com.github.devcsrj.klerk.Chamber
import com.github.devcsrj.klerk.Congress
import com.github.devcsrj.klerk.Journal
import com.github.devcsrj.klerk.Session
import org.apache.beam.sdk.transforms.DoFn
import org.slf4j.LoggerFactory
import java.io.File
import java.net.URI
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.time.format.TextStyle
import java.util.*
import java.util.function.Predicate

/**
 * Strips off the heading and footer content written to each file
 */
internal class StripBorderText : DoFn<File, File>() {

    private val logger = LoggerFactory.getLogger(StripBorderText::class.java)

    private val objectMapper = ObjectMapper()

    @ProcessElement
    fun processElement(
        @Element file: File,
        outputReceiver: OutputReceiver<File>
    ) {

        val number = file.nameWithoutExtension.substringAfter("journal-")
        val json = file.parentFile.resolve("journal-$number.json")
        val journal = readJournal(json)
        logger.info("✂️ $journal")

        val shouldSkip = isDateHeading(journal).or(isSessionHeading(journal))
        val txt = file.parentFile.resolve("journal-$number.txt")
        val tmp = file.parentFile.resolve("journal-$number.tmp")
        tmp.bufferedWriter().use { writer ->
            txt.bufferedReader().lineSequence()
                .filter { !shouldSkip.test(it) }
                .forEach {
                    writer.write(it)
                    writer.write(System.lineSeparator())
                }
        }
        Files.move(tmp.toPath(), txt.toPath(), StandardCopyOption.REPLACE_EXISTING)

        outputReceiver.output(txt)
    }

    private fun readJournal(json: File): Journal {
        val map = objectMapper.readValue(json, Map::class.java) as Map<String, Any>
        val chamber = Chamber.valueOf(map["chamber"] as String)
        val congress = Congress(map["congress"] as Int)
        val session = (map["session"] as Map<String, Any>).let {
            Session(it["number"] as Int, Session.Type.valueOf(it["type"] as String))
        }
        val number = map["number"] as Int
        val date = LocalDate.parse(
            map["date"] as String,
            DateTimeFormatter.ofPattern("yyyy-MM-dd")
        )
        val uri = URI.create(map["document_uri"] as String)
        return Journal(
            chamber = chamber,
            congress = congress,
            session = session,
            number = number,
            date = date,
            documentUri = uri
        )
    }

    private fun isDateHeading(journal: Journal): Predicate<String> {
        return Predicate {
            val dow = journal.date.dayOfWeek.getDisplayName(TextStyle.FULL, Locale.ENGLISH)
            val month = journal.date.month.getDisplayName(TextStyle.FULL, Locale.ENGLISH)
            val str = "$dow, $month"
            // JOURNAL NO. 18 Tuesday, September 11, 2018
            it.contains(str) || it.contains(str.toUpperCase())
        }
    }

    private fun isSessionHeading(journal: Journal): Predicate<String> {
        return Predicate {
            val prefix = journal.congress.toString()
            // 2  17th Congress 3rd Regular Session
            it.contains(prefix) && it.contains("Session")
        }
    }
}