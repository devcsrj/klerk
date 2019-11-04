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

import com.github.devcsrj.klerk.*
import org.apache.beam.sdk.testing.PAssert
import org.apache.beam.sdk.testing.TestPipeline
import org.apache.beam.sdk.transforms.Create
import org.apache.beam.sdk.transforms.GroupByKey
import org.apache.beam.sdk.transforms.ParDo
import org.apache.beam.sdk.values.KV
import org.spekframework.spek2.Spek
import java.net.URI
import java.time.LocalDate
import java.time.Month
import kotlin.test.assertEquals

object PipelineTest : Spek({

    test("17th Senate R3 J3", timeout = 60 * 10000) {

        val journal = Journal(
            congress = Congress(17),
            chamber = Chamber.SENATE,
            session = Session.regular(3),
            number = 3,
            date = LocalDate.of(2018, Month.JULY, 25),
            documentUri = URI("http://www.senate.gov.ph/lisdata/2822824543!.pdf")
        )
        val maxPages = 2
        val files = mutableListOf<KV<Journal, Page>>()
        for (i in 0..maxPages) {
            val file = javaClass.copyTempResource("/journal/17th-s-r3/journal-3-p$i.png")
            val page = Page(i, file)
            files.add(KV.of(journal, page))
        }

        val pipeline = TestPipeline.create().apply {
            enableAbandonedNodeEnforcement(false)
            coderRegistry.apply {
                registerCoderForClass(Journal::class.java, FstCoder<Journal>())
            }
        }

        val output = pipeline
            .apply("List", Create.of(files))
            .apply("Slice", ParDo.of(SliceSection()))
            .apply("Deskew", ParDo.of(DeskewSection()))
            .apply("Skip", ParDo.of(SkipSection()))
            .apply("GroupSlices", GroupByKey.create())
            .apply("Read", ParDo.of(ReadSection()))
            .apply("GroupBlocks", GroupByKey.create())
            .apply("Write", ParDo.of(WriteJournal()))


        val first = files.iterator().next().value
        val txt = first.file.parentFile.resolve("journal-${journal.number}.txt")
        txt.deleteOnExit()
        PAssert
            .that(output)
            .containsInAnyOrder(txt)

        pipeline.run()

        val resource = "/journal/17th-s-r3/journal-3.txt"
        javaClass.getResourceAsStream(resource).use { src ->
            src.bufferedReader().use {
                assertEquals(it.readText(), txt.readText())
            }
        }
    }
})