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