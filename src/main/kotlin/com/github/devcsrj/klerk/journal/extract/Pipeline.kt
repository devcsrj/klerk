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

import org.apache.beam.sdk.Pipeline
import org.apache.beam.sdk.options.PipelineOptionsFactory
import org.apache.beam.sdk.transforms.Create
import org.apache.beam.sdk.transforms.ParDo
import java.io.File

fun main(args: Array<String>) {
    val options = PipelineOptionsFactory
        .fromArgs(*args)
        .withValidation()
        .`as`(Options::class.java)

    val regex = Regex("journal-\\d+\\.pdf\$")
    val src = File(options.getDir())
        .walkTopDown()
        .filter { it.name.matches(regex) }
        .toList()

    val pipeline = Pipeline.create(options)

    pipeline
        .apply("List", Create.of(src))
        .apply("ToImage", ParDo.of(PdfToImage()))
        .apply("Crop", ParDo.of(CropContent()))

    pipeline.run().waitUntilFinish()
}