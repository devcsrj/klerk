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