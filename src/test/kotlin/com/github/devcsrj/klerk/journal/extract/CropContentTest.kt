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

import org.apache.beam.sdk.testing.PAssert
import org.apache.beam.sdk.testing.TestPipeline
import org.apache.beam.sdk.transforms.Create
import org.apache.beam.sdk.transforms.ParDo
import org.spekframework.spek2.Spek
import java.awt.Dimension
import java.nio.file.Files
import kotlin.test.assertEquals

object CropContentTest : Spek({

    group("bordered images") {
        /**
         * Path => expected cropped dimension
         */
        val resources: Map<String, Dimension> = mapOf(
            "17th-h-r2-j28-p2.png" to Dimension(2228, 3180),
            "17th-s-r1-j72-p2.png" to Dimension(2340, 3119),
            "17th-s-r1-j72-p1.png" to Dimension(2355, 3111),
            "17th-s-r3-j16-p21.png" to Dimension(1183, 1472)
        )

        resources.forEach { (resource, expected) ->
            test("Crop $resource") {
                val png = "/journal/crop/$resource"
                val prefix = resource.substringBeforeLast('.')
                val original = Files.createTempFile(prefix, ".png").toFile()
                original.outputStream().use { sink ->
                    javaClass.getResourceAsStream(png).use { src ->
                        src.copyTo(sink)
                    }
                }
                original.deleteOnExit()

                val filename = "${original.nameWithoutExtension}-cropped.png"
                val cropped = original.parentFile.resolve(filename)
                cropped.deleteOnExit()

                val pipeline = TestPipeline.create()
                    .enableAbandonedNodeEnforcement(false)
                val input = pipeline.apply(Create.of(original))
                val output = input.apply(ParDo.of(CropContent()))

                PAssert
                    .that(output)
                    .containsInAnyOrder(cropped)

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
            val png = "/journal/crop/$resource"
            val prefix = resource.substringBeforeLast('.')
            val original = Files.createTempFile(prefix, ".png").toFile()
            original.outputStream().use { sink ->
                javaClass.getResourceAsStream(png).use { src ->
                    src.copyTo(sink)
                }
            }
            original.deleteOnExit()

            val pipeline = TestPipeline.create()
                .enableAbandonedNodeEnforcement(false)
            val input = pipeline.apply(Create.of(original))
            val output = input.apply(ParDo.of(CropContent()))

            PAssert
                .that(output)
                .containsInAnyOrder(original)
            pipeline.run()
        }
    }
})