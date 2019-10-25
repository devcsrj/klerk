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