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
import org.apache.beam.sdk.Pipeline
import org.apache.beam.sdk.options.Description
import org.apache.beam.sdk.options.PipelineOptions
import org.apache.beam.sdk.options.PipelineOptionsFactory
import org.apache.beam.sdk.options.Validation
import org.apache.beam.sdk.transforms.Create
import org.apache.beam.sdk.transforms.DoFn
import org.apache.beam.sdk.transforms.ParDo
import org.slf4j.LoggerFactory
import java.nio.file.Path


class JournalPipeline {

    companion object {

        private val logger = LoggerFactory.getLogger(JournalPipeline::class.java)
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
                    logger.info("✒️ $journal")
                    outputReceiver.output(journal)
                }

                val ss = senateApi.fetch(congress, session)
                for (journal in ss) {
                    logger.info("✒️ $journal")
                    outputReceiver.output(journal)
                }
            }
        }
    }

    interface Options : PipelineOptions {

        @Description("The congress # to fetch journals from (e.g., 17, 18)")
        @Validation.Required
        fun getInput(): List<Int>

        fun setInput(input: List<Int>)

        @Description("The destination directory to write the files to")
        fun getOutput(): Path

        fun setOutput(output: Path)
    }
}

fun main(args: Array<String>) {
    val options = PipelineOptionsFactory
        .fromArgs(*args)
        .withValidation()
        .`as`(JournalPipeline.Options::class.java)
    val congresses = Create.of(options.getInput()
        .map { Congress(it) }
        .toList())

    val pipeline = Pipeline.create(options)

    pipeline
        .apply("Prepare", congresses)
        .apply("Fetch", ParDo.of(JournalPipeline.Fetch()))

    pipeline.run().waitUntilFinish()
}