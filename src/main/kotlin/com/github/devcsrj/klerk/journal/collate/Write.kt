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
package com.github.devcsrj.klerk.journal.collate

import com.github.devcsrj.klerk.Journal
import com.github.devcsrj.klerk.journal.directoryFor
import org.apache.beam.sdk.transforms.DoFn
import java.io.File
import java.time.format.DateTimeFormatter

/**
 * Writes the journal metadata to a local directory
 */
internal class Write(private val dist: File) : DoFn<Journal, Journal>() {

    init {
        require(dist.isDirectory) {
            "Expecting a directory, but got $dist"
        }
    }

    @ProcessElement
    fun processElement(
        @Element journal: Journal,
        outputReceiver: OutputReceiver<Journal>
    ) {

        val format = DateTimeFormatter.ofPattern("yyyy-MM-dd")
        val dir = directoryFor(dist, journal)
        dir.mkdirs()

        val json = dir.resolve("journal-${journal.number}.json")
        if (!json.exists()) {
            json.writeText(
                """
                {
                    "congress": ${journal.congress.number},
                    "chamber": "${journal.chamber}",
                    "session": {
                        "number": ${journal.session.number},
                        "type": "${journal.session.type}"
                    },
                    "number": ${journal.number},
                    "date": "${format.format(journal.date)}",
                    "document_uri": "${journal.documentUri}"
                }
                """.trimIndent()
            )
        }

        outputReceiver.output(journal)
    }
}