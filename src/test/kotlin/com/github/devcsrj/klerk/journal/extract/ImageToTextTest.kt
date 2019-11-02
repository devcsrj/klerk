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

import com.github.devcsrj.klerk.Chamber
import com.github.devcsrj.klerk.Congress
import com.github.devcsrj.klerk.Journal
import com.github.devcsrj.klerk.Session
import org.apache.beam.sdk.testing.PAssert
import org.apache.beam.sdk.testing.TestPipeline
import org.apache.beam.sdk.transforms.*
import org.apache.beam.sdk.values.KV
import org.spekframework.spek2.Spek
import java.io.File
import java.net.URI
import java.time.LocalDate
import java.time.Month
import kotlin.test.assertEquals

object ImageToTextTest : Spek({

    val dir = File(System.getProperty("java.io.tmpdir"))

    test("Extract text from Senate R3 Journal 3", timeout = 30 * 1000) {

        val journal = Journal(
            chamber = Chamber.SENATE,
            congress = Congress(17),
            session = Session.regular(3),
            number = 3,
            date = LocalDate.of(2019, Month.OCTOBER, 29),
            documentUri = URI.create("https://example.com")
        )
        val theirJournals = object : SimpleFunction<Page, KV<Journal, Page>>() {
            override fun apply(input: Page) = KV.of(journal, input)
        }

        val resources = listOf(
            "17th-s-r3-j3-p0.png",
            "17th-s-r3-j3-p1.png",
            "17th-s-r3-j3-p2.png"
        )

        val pages = mutableListOf<Page>()
        for ((i, resource) in resources.withIndex()) {
            val tmp = dir.resolve(resource)
            val png = "/journal/image-to-text/$resource"
            tmp.outputStream().use { sink ->
                javaClass.getResourceAsStream(png).use { src ->
                    src.copyTo(sink)
                }
            }
            tmp.deleteOnExit()
            pages.add(Page(i, tmp))
        }

        val pipeline = TestPipeline.create()
            .enableAbandonedNodeEnforcement(false)
        pipeline.coderRegistry.apply {
            registerCoderForClass(Journal::class.java, FstCoder<Journal>())
        }
        val output = pipeline
            .apply(Create.of(pages.shuffled()))
            .apply(MapElements.via(theirJournals))
            .apply("GroupPagesByJournal", GroupByKey.create())
            .apply(ParDo.of(DetectSections()))
            .apply("GroupSectionsByJournal", GroupByKey.create())
            .apply(ParDo.of(ImageToText()))

        val txt = dir.resolve("journal-${journal.number}.txt")
        val expected = KV.of(journal, txt)
        PAssert
            .that(output)
            .containsInAnyOrder(expected)

        pipeline.run()

        val resource = "/journal/image-to-text/17th-s-r3-j3.txt"
        javaClass.getResourceAsStream(resource).use { src ->
            src.bufferedReader().use {
                assertEquals(it.readText(), txt.readText())
            }
        }
    }

    test("Extract text from House R3 Journal 51", timeout = 30 * 1000) {

        val journal = Journal(
            chamber = Chamber.HOUSE,
            congress = Congress(17),
            session = Session.regular(3),
            number = 51,
            date = LocalDate.of(2019, Month.OCTOBER, 29),
            documentUri = URI.create("https://example.com")
        )
        val theirJournals = object : SimpleFunction<Page, KV<Journal, Page>>() {
            override fun apply(input: Page) = KV.of(journal, input)
        }

        val resources = listOf(
            "17th-h-r3-j51-p3.png"
        )

        val pages = mutableListOf<Page>()
        for ((i, resource) in resources.withIndex()) {
            val tmp = dir.resolve(resource)
            val png = "/journal/image-to-text/$resource"
            tmp.outputStream().use { sink ->
                javaClass.getResourceAsStream(png).use { src ->
                    src.copyTo(sink)
                }
            }
            tmp.deleteOnExit()
            pages.add(Page(i, tmp))
        }

        val pipeline = TestPipeline.create()
            .enableAbandonedNodeEnforcement(false)
        pipeline.coderRegistry.apply {
            registerCoderForClass(Journal::class.java, FstCoder<Journal>())
        }
        val output = pipeline
            .apply(Create.of(pages.shuffled()))
            .apply(MapElements.via(theirJournals))
            .apply("GroupPagesByJournal", GroupByKey.create())
            .apply(ParDo.of(DetectSections()))
            .apply("GroupSectionsByJournal", GroupByKey.create())
            .apply(ParDo.of(ImageToText()))

        val txt = dir.resolve("journal-${journal.number}.txt")
        val expected = KV.of(journal, txt)
        PAssert
            .that(output)
            .containsInAnyOrder(expected)

        pipeline.run()

        println(txt.readText())
        val resource = "/journal/image-to-text/17th-h-r3-j51.txt"
        javaClass.getResourceAsStream(resource).use { src ->
            src.bufferedReader().use {
                assertEquals(it.readText(), txt.readText())
            }
        }
    }
})