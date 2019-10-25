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

import com.github.devcsrj.klerk.Journal
import com.github.devcsrj.klerk.journal.fromJson
import org.apache.beam.sdk.transforms.DoFn
import org.joda.time.LocalDate
import java.io.File
import java.util.*

/**
 * Outputs each file, with the journal's publication date as
 * timestamp.
 *
 * This is mainly used for windowing.
 */
internal class AddJournalTimestamp : DoFn<File, File>() {

    @ProcessElement
    fun processElement(
        @Element file: File,
        outputReceiver: OutputReceiver<File>
    ) {
        val name = file.nameWithoutExtension
        val json = file.parentFile.resolve("$name.json")
        val journal = Journal.fromJson(json.readText())

        val localDate = journal.date.let {
            val d = Date(it.toEpochDay())
            LocalDate.fromDateFields(d)
        }
        val timestamp = localDate.toDateTimeAtStartOfDay().toInstant()
        outputReceiver.outputWithTimestamp(file, timestamp)
    }
}