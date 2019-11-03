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
import java.awt.Dimension
import java.io.File
import java.net.URI
import java.time.LocalDate
import java.time.Month
import kotlin.test.assertEquals

object CropTest : Spek({

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
    val dir = File(System.getProperty("java.io.tmpdir"))
    val filename = "journal-${journal.number}-p1-cropped.png"
    val cropped = dir.resolve(filename)

    beforeEachTest {
        cropped.delete()
    }

    group("bordered images") {

        /**
         * Path => expected cropped dimension
         */
        val resources: Map<String, Dimension> = mapOf(
            "17th-h-r2-j28-p2.png" to Dimension(2224, 3194),
            "17th-s-r1-j72-p2.png" to Dimension(2336, 3133),
            "17th-s-r1-j72-p1.png" to Dimension(2351, 3126),
            "17th-s-r3-j16-p21.png" to Dimension(1132, 1488),
            "17th-s-r3-j37-p24.png" to Dimension(1146, 1517)
        )

        resources.forEach { (resource, expected) ->
            test("Crop $resource", timeout = 15 * 1000) {
                val png = "/journal/crop/$resource"
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
                val output = input.apply(ParDo.of(Crop()))

                PAssert
                    .that(output)
                    .containsInAnyOrder(KV.of(journal, Page(1, cropped)))

                pipeline.run()

                val actual = Images.dimensionOf(cropped)
                assertEquals(expected, actual)
            }
        }
    }

    group("borderless images") {

        val resources: List<String> = listOf(
            "17th-h-r2-j28-p1.png",
            "17th-s-r1-j72-p0.png"
        )

        resources.forEach { resource ->

            test("Crop $resource") {
                val png = "/journal/crop/$resource"
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
                val output = input.apply(ParDo.of(Crop()))

                PAssert
                    .that(output)
                    .containsInAnyOrder(KV.of(journal, Page(1, original)))
                pipeline.run()
            }
        }
    }
})