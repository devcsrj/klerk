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
import org.apache.beam.sdk.transforms.Create
import org.apache.beam.sdk.transforms.MapElements
import org.apache.beam.sdk.transforms.ParDo
import org.apache.beam.sdk.transforms.SimpleFunction
import org.apache.beam.sdk.values.KV
import org.spekframework.spek2.Spek
import java.io.File
import java.net.URI
import java.time.LocalDate
import java.time.Month

object DeskewTest : Spek({

    val journal = Journal(
        chamber = Chamber.SENATE,
        congress = Congress(17),
        session = Session.regular(1),
        number = 1,
        date = LocalDate.of(2019, Month.OCTOBER, 29),
        documentUri = URI.create("https://example.com")
    )
    val asPage = object : SimpleFunction<File, KV<Journal, Page>>() {
        override fun apply(input: File) = KV.of(journal, Page(1, input))
    }

    group("skewed images") {

        val resources: List<String> = listOf(
            "17th-h-r3-j31-p13.png",
            "17th-s-r3-j12-p0.png",
            "17th-s-r3-j3-p1.png",
            "17th-s-r3-j1-p16.png",
            "17th-s-r3-j61-p48.png",
            "17th-s-r3-j1-p20.png",
            "17th-s-r3-j42-p0.png"
        )

        val dir = File(System.getProperty("java.io.tmpdir"))
        val filename = "journal-${journal.number}-p1-deskewed.png"
        val deskewed = dir.resolve(filename)

        beforeEachTest {
            deskewed.delete()
        }

        resources.forEach { resource ->

            test("Deskew $resource", timeout = 30 * 1000) {
                val png = "/journal/deskew/$resource"
                val prefix = resource.substringBeforeLast('.')
                val original = dir.resolve("$prefix.png")
                original.outputStream().use { sink ->
                    javaClass.getResourceAsStream(png).use { src ->
                        src.copyTo(sink)
                    }
                }
                original.deleteOnExit()

                val pipeline = TestPipeline.create()
                    .enableAbandonedNodeEnforcement(false)
                val input = pipeline
                    .apply(Create.of(original))
                    .apply(MapElements.via(asPage))
                val output = input.apply(ParDo.of(Deskew()))

                PAssert
                    .that(output)
                    .containsInAnyOrder(KV.of(journal, Page(1, deskewed)))

                pipeline.run()

                // Don't know how to assert content without scanning again println()
            }
        }
    }
})