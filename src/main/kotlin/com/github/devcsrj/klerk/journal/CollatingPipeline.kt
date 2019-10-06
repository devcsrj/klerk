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
package com.github.devcsrj.klerk.journal

import com.github.devcsrj.klerk.Congress
import com.github.devcsrj.klerk.Journal
import com.github.devcsrj.klerk.Session
import com.github.devcsrj.klerk.downloadTo
import org.apache.beam.sdk.Pipeline
import org.apache.beam.sdk.options.Description
import org.apache.beam.sdk.options.PipelineOptions
import org.apache.beam.sdk.options.PipelineOptionsFactory
import org.apache.beam.sdk.options.Validation
import org.apache.beam.sdk.transforms.Create
import org.apache.beam.sdk.transforms.DoFn
import org.apache.beam.sdk.transforms.ParDo
import org.slf4j.LoggerFactory
import java.io.File
import java.time.format.DateTimeFormatter


class CollatingPipeline {

    companion object {

        private val logger = LoggerFactory.getLogger(CollatingPipeline::class.java)

    }

    /**
     * Retrieves all [Session]s from each Congress of both the house and the senate
     */
    class Fetch : DoFn<Congress, Journal>() {

        private val sessions: Array<Session> = arrayOf(
            Session.regular(1),
            Session.regular(2),
            Session.regular(3)
        )

        @ProcessElement
        fun processElement(
            @Element congress: Congress,
            outputReceiver: OutputReceiver<Journal>
        ) {

            val houseApi: JournalApi = HouseHttpJournalApi()
            val senateApi: JournalApi = SenateHttpJournalApi()

            for (session in sessions) {
                val hs = houseApi.fetch(congress, session)
                for (journal in hs) {
                    logger.info("📄️ $journal")
                    outputReceiver.output(journal)
                }

                val ss = senateApi.fetch(congress, session)
                for (journal in ss) {
                    logger.info("📄️ $journal")
                    outputReceiver.output(journal)
                }
            }
        }
    }

    /**
     * Writes the journal metadata to a local directory
     */
    class Write(private val dist: File) : DoFn<Journal, Journal>() {

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

            val format = DateTimeFormatter.ofPattern("YYYY-MM-dd")
            val dir = directoryFor(dist, journal)
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

    /**
     * Downloads [Journal.documentUri] to the local directory
     */
    class Download(private val dist: File) : DoFn<Journal, Journal>() {

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

    interface Options : PipelineOptions {

        @Description("The congress # to fetch journals from (e.g., 17, 18)")
        @Validation.Required
        fun getInput(): List<Int>

        fun setInput(input: List<Int>)

        @Description("The destination directory to write the files to")
        fun getOutput(): String

        fun setOutput(output: String)
    }
}

fun main(args: Array<String>) {
    val options = PipelineOptionsFactory
        .fromArgs(*args)
        .withValidation()
        .`as`(CollatingPipeline.Options::class.java)

    val dist = File(options.getOutput())
    dist.mkdirs()

    val congresses = Create.of(options.getInput()
        .map { Congress(it) }
        .toList())

    val pipeline = Pipeline.create(options)

    pipeline
        .apply("Prepare", congresses)
        .apply("Fetch", ParDo.of(CollatingPipeline.Fetch()))
        .apply("Write", ParDo.of(CollatingPipeline.Write(dist)))
        .apply("Download", ParDo.of(CollatingPipeline.Download(dist)))

    pipeline.run().waitUntilFinish()
}