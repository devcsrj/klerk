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
package com.github.devcsrj.klerk.journal.collate

import com.github.devcsrj.klerk.Congress
import com.github.devcsrj.klerk.Journal
import org.apache.beam.sdk.Pipeline
import org.apache.beam.sdk.options.PipelineOptionsFactory
import org.apache.beam.sdk.transforms.Create
import org.apache.beam.sdk.transforms.ParDo
import org.apache.beam.sdk.transforms.windowing.FixedWindows
import org.apache.beam.sdk.transforms.windowing.Window
import org.joda.time.Duration
import java.io.File

fun main(args: Array<String>) {
    val options = PipelineOptionsFactory
        .fromArgs(*args)
        .withValidation()
        .`as`(Options::class.java)

    val dist = File(options.getOutput())
    dist.mkdirs()

    val congresses = Create.of(options.getInput()
        .map { Congress(it) }
        .toList())

    val pipeline = Pipeline.create(options)

    pipeline
        .apply("Prepare", congresses)
        .apply("Fetch", ParDo.of(Fetch()))
        .apply(Window.into<Journal>(FixedWindows.of(Duration.standardDays(5))))
        .apply("Write", ParDo.of(Write(dist)))
        .apply("Download", ParDo.of(Download(dist)))

    pipeline.run().waitUntilFinish()
}