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
import org.spekframework.spek2.style.gherkin.Feature
import java.io.File
import java.nio.file.Files

object DeskewContentTest : Spek({

    Feature("deskew senate journal") {

        Scenario("a content page") {

            val png = "/journal/deskew/17th-h-r3-j31-p13.png"
            lateinit var original: File

            Given("17th-h-r3-j31-p13") {
                original = Files.createTempFile("", ".png").toFile()
                original.outputStream().use { sink ->
                    javaClass.getResourceAsStream(png).use { src ->
                        src.copyTo(sink)
                    }
                }
                original.deleteOnExit()
            }

            When("processed", timeout = 60 * 1000) {
                val pipeline = TestPipeline.create()
                    .enableAbandonedNodeEnforcement(false)
                val input = pipeline.apply(Create.of(original))
                val output = input.apply(ParDo.of(DeskewContent()))

                PAssert
                    .that(output)
                    .containsInAnyOrder(original)

                pipeline.run()
            }

            Then("content is deskewed") {
                // Don't know how to assert that content is skewed without
                // loading the image again...
            }
        }

        Scenario("a title page") {

            val png = "/journal/deskew/17th-s-r3-j12-p0.png"
            lateinit var original: File

            Given("17th-s-r3-j12-p0") {
                original = Files.createTempFile("", ".png").toFile()
                original.outputStream().use { sink ->
                    javaClass.getResourceAsStream(png).use { src ->
                        src.copyTo(sink)
                    }
                }
                original.deleteOnExit()
            }

            When("processed", timeout = 60 * 1000) {
                val pipeline = TestPipeline.create()
                    .enableAbandonedNodeEnforcement(false)
                val input = pipeline.apply(Create.of(original))
                val output = input.apply(ParDo.of(DeskewContent()))

                PAssert
                    .that(output)
                    .containsInAnyOrder(original)

                pipeline.run()
            }

            Then("content is deskewed") {
                // Don't know how to assert that content is skewed without
                // loading the image again...
            }
        }

        Scenario("a start page") {

            val png = "/journal/deskew/17th-s-r3-j3-p1.png"
            lateinit var original: File

            Given("17th-s-r3-j3-p1.png") {
                original = Files.createTempFile("", ".png").toFile()
                original.outputStream().use { sink ->
                    javaClass.getResourceAsStream(png).use { src ->
                        src.copyTo(sink)
                    }
                }
                original.deleteOnExit()
            }

            When("processed", timeout = 60 * 1000) {
                val pipeline = TestPipeline.create()
                    .enableAbandonedNodeEnforcement(false)
                val input = pipeline.apply(Create.of(original))
                val output = input.apply(ParDo.of(DeskewContent()))

                PAssert
                    .that(output)
                    .containsInAnyOrder(original)

                pipeline.run()
            }

            Then("content is deskewed") {
                // Don't know how to assert that content is skewed without
                // loading the image again...
            }
        }
    }
})