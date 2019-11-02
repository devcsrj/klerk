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

import com.github.devcsrj.klerk.Journal
import org.apache.beam.sdk.transforms.DoFn
import org.apache.beam.sdk.values.KV
import org.bytedeco.leptonica.global.lept.pixDestroy
import org.bytedeco.leptonica.global.lept.pixRead
import org.bytedeco.tesseract.ETEXT_DESC
import org.bytedeco.tesseract.TessBaseAPI
import org.bytedeco.tesseract.global.tesseract.RIL_PARA
import org.slf4j.LoggerFactory
import java.io.File

/**
 * Loads the [Page.file] as images, and extracts text.
 *
 * The extracted text is written to disk, under the filename
 * `journal-{number}.txt`
 */
internal class ImageToText : DoFn<KV<Journal, Iterable<@JvmWildcard PageSection>>, KV<Journal, File>>() {

    private val logger = LoggerFactory.getLogger(ImageToText::class.java)

    @ProcessElement
    fun processElement(context: ProcessContext) {

        val tesseract = TessBaseAPI()
        if (tesseract.Init(".", "eng") != 0) {
            logger.error("⚠️ Could not find OCR training data 'eng.traineddata'")
            return
        }

        val element = context.element()
        val journal = element.key!!
        val firstPage = element.value.iterator().next()
        val name = "journal-${journal.number}.txt"
        val outputFile = firstPage.file.parentFile.resolve(name)

        logger.info("🔎 Journal ${journal.number}")

        outputFile.printWriter().use { writer ->
            val sections = element.value.groupBy { it.page }.toSortedMap()
            tesseract.use { api ->
                for ((k, v) in sections) {
                    val file = v.first().file
                    val paragraphs = readSection(file, v, api)
                    for (p in paragraphs) {
                        writer.println(p)
                    }
                }
            }
        }
        context.output(KV.of(journal, outputFile))
    }

    private fun readSection(
        file: File,
        sections: Iterable<PageSection>,
        api: TessBaseAPI
    ): Sequence<String> {

        return sequence {
            val items = sections.sorted()
            val image = pixRead(file.toString())
            api.SetImage(image)
            api.SetVariable("preserve_interword_spaces", "1")
            for (item in items) {
                val rect = item.rectangle
                api.SetRectangle(rect.x, rect.y, rect.width, rect.height)
                api.Recognize(ETEXT_DESC())

                val iterator = api.GetIterator()
                val level = RIL_PARA

                iterator.Begin()
                do {
                    val ptr = iterator.GetUTF8Text(level)
                    if (ptr != null)
                        yield(ptr.getString("UTF-8"))
                } while (iterator.Next(level))
            }
            pixDestroy(image)
        }
    }
}