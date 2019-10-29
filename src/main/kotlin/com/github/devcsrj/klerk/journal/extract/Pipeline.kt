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
        .apply("LoadInfo", ParDo.of(LoadJournalInfo()))
        .apply("ToImage", ParDo.of(PdfToImage()))
        .apply("Deskew", ParDo.of(DeskewContent()))
        .apply("Crop", ParDo.of(CropContent()))

    pipeline.run().waitUntilFinish()
}