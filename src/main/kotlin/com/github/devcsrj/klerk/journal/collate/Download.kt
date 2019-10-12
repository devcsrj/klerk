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
import com.github.devcsrj.klerk.downloadTo
import com.github.devcsrj.klerk.journal.directoryFor
import org.apache.beam.sdk.transforms.DoFn
import org.slf4j.LoggerFactory
import java.io.File

/**
 * Downloads [Journal.documentUri] to the local directory
 */
internal class Download(private val dist: File) : DoFn<Journal, Journal>() {

    private val logger = LoggerFactory.getLogger(Download::class.java)

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

        val dir = directoryFor(dist, journal)
        val pdf = dir.resolve("journal-${journal.number}.pdf")
        logger.info("⬇ Journal ${journal.number} - ${journal.documentUri}")
        if (!pdf.exists()) {
            journal.documentUri.toURL().downloadTo(pdf.toPath())
        }

        outputReceiver.output(journal)
    }
}