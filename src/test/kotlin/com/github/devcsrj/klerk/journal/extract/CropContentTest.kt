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
import java.awt.Dimension
import java.io.File
import java.nio.file.Files
import kotlin.test.assertEquals

object CropContentTest : Spek({

    Feature("crop house journal") {

        Scenario("page has border") {

            val png = "/journal/crop/17th-h-r2-j28-p2.png"
            lateinit var original: File
            lateinit var cropped: File

            Given("17th-h-r2-j28-p2") {
                original = Files.createTempFile("", ".png").toFile()
                original.outputStream().use { sink ->
                    javaClass.getResourceAsStream(png).use { src ->
                        src.copyTo(sink)
                    }
                }
                original.deleteOnExit()

                val filename = "${original.nameWithoutExtension}-cropped.png"
                cropped = original.parentFile.resolve(filename)
                cropped.deleteOnExit()
            }

            When("processed", timeout = 60 * 1000) {
                val pipeline = TestPipeline.create()
                    .enableAbandonedNodeEnforcement(false)
                val input = pipeline.apply(Create.of(original))
                val output = input.apply(ParDo.of(CropContent()))

                PAssert
                    .that(output)
                    .containsInAnyOrder(cropped)

                pipeline.run()
            }

            Then("content is cropped") {
                val expected = Dimension(2192, 3180)
                val actual = Images.dimensionOf(cropped)
                assertEquals(expected, actual)
            }
        }

        Scenario("page has no border") {

            val png = "/journal/crop/17th-h-r2-j28-p1.png"
            lateinit var original: File

            Given("17th-h-r2-j28-p1") {
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
                val output = input.apply(ParDo.of(CropContent()))

                PAssert
                    .that(output)
                    .containsInAnyOrder(original)
                pipeline.run()
            }

            Then("nothing is cropped") {
                // nothing
            }
        }
    }

    Feature("crop senate journal") {

        Scenario("page has border") {

            val png = "/journal/crop/17th-s-r1-j72-p2.png"
            lateinit var original: File
            lateinit var cropped: File

            Given("17th-s-r1-j72-p2.png") {
                original = Files.createTempFile("", ".png").toFile()
                original.outputStream().use { sink ->
                    javaClass.getResourceAsStream(png).use { src ->
                        src.copyTo(sink)
                    }
                }
                original.deleteOnExit()

                val filename = "${original.nameWithoutExtension}-cropped.png"
                cropped = original.parentFile.resolve(filename)
                cropped.deleteOnExit()
            }

            When("processed", timeout = 60 * 1000) {
                val pipeline = TestPipeline.create()
                    .enableAbandonedNodeEnforcement(false)
                val input = pipeline.apply(Create.of(original))
                val output = input.apply(ParDo.of(CropContent()))

                PAssert
                    .that(output)
                    .containsInAnyOrder(cropped)

                pipeline.run()
            }

            Then("content is cropped") {
                val expected = Dimension(2241, 3119)
                val actual = Images.dimensionOf(cropped)
                assertEquals(expected, actual)
            }
        }

        Scenario("page has border and large whitespace") {

            val png = "/journal/crop/17th-s-r1-j72-p1.png"
            lateinit var original: File
            lateinit var cropped: File

            Given("17th-s-r1-j72-p1.png") {
                original = Files.createTempFile("", ".png").toFile()
                original.outputStream().use { sink ->
                    javaClass.getResourceAsStream(png).use { src ->
                        src.copyTo(sink)
                    }
                }
                original.deleteOnExit()

                val filename = "${original.nameWithoutExtension}-cropped.png"
                cropped = original.parentFile.resolve(filename)
                cropped.deleteOnExit()
            }

            When("processed", timeout = 60 * 1000) {
                val pipeline = TestPipeline.create()
                    .enableAbandonedNodeEnforcement(false)
                val input = pipeline.apply(Create.of(original))
                val output = input.apply(ParDo.of(CropContent()))

                PAssert
                    .that(output)
                    .containsInAnyOrder(cropped)

                pipeline.run()
            }

            Then("content is cropped") {
                val expected = Dimension(2251, 3111)
                val actual = Images.dimensionOf(cropped)
                assertEquals(expected, actual)
            }
        }

        Scenario("page has no border") {

            val png = "/journal/crop/17th-s-r1-j72-p0.png"
            lateinit var original: File

            Given("17th-s-r1-j72-p0.png") {
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
                val output = input.apply(ParDo.of(CropContent()))

                PAssert
                    .that(output)
                    .containsInAnyOrder(original)
                pipeline.run()
            }

            Then("nothing is cropped") {
                // nothing
            }
        }
    }
})