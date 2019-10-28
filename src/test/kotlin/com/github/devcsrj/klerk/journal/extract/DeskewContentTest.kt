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
import java.nio.file.Files

object DeskewContentTest : Spek({

    group("skewed images") {

        val resources: List<String> = listOf(
            "17th-h-r3-j31-p13.png",
            "17th-s-r3-j12-p0.png",
            "17th-s-r3-j3-p1.png",
            "17th-s-r3-j1-p16.png",
            "17th-s-r3-j61-p48.png"
        )

        resources.forEach { resource ->

            test("Deskew $resource", timeout = 30 * 1000) {
                val png = "/journal/deskew/$resource"
                val prefix = resource.substringBeforeLast('.')
                val original = Files.createTempFile(prefix, ".png").toFile()
                original.outputStream().use { sink ->
                    javaClass.getResourceAsStream(png).use { src ->
                        src.copyTo(sink)
                    }
                }
                original.deleteOnExit()

                val filename = "${original.nameWithoutExtension}-deskewed.png"
                val deskewed = original.parentFile.resolve(filename)
                deskewed.deleteOnExit()

                val pipeline = TestPipeline.create()
                    .enableAbandonedNodeEnforcement(false)
                val input = pipeline.apply(Create.of(original))
                val output = input.apply(ParDo.of(DeskewContent()))

                PAssert
                    .that(output)
                    .containsInAnyOrder(deskewed)

                pipeline.run()

                // Don't know how to assert content without scanning again println()
            }
        }
    }
})