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
import org.bytedeco.opencv.global.opencv_core.BORDER_CONSTANT
import org.bytedeco.opencv.global.opencv_core.CV_8UC1
import org.bytedeco.opencv.global.opencv_imgcodecs.imread
import org.bytedeco.opencv.global.opencv_imgproc.*
import org.bytedeco.opencv.opencv_core.Mat
import org.bytedeco.opencv.opencv_core.MatVector
import org.bytedeco.opencv.opencv_core.Point
import org.bytedeco.opencv.opencv_core.Rect
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
internal class ImageToText : DoFn<KV<Journal, Iterable<@JvmWildcard Page>>, KV<Journal, File>>() {

    private val logger = LoggerFactory.getLogger(ImageToText::class.java)

    @ProcessElement
    fun processElement(context: ProcessContext) {

        val api = TessBaseAPI()
        if (api.Init(".", "eng") != 0) {
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
            api.use {
                val paragraphs = readParagraphs(element.value, it)
                for (p in paragraphs) {
                    writer.println(p)
                }
            }
        }
        context.output(KV.of(journal, outputFile))
    }

    private fun readParagraphs(pages: Iterable<Page>, api: TessBaseAPI): Sequence<String> {
        return pages
            .sorted()
            .map { readParagraphs(it, api) }
            .reduce { acc, sequence -> acc + sequence }
    }

    private fun readParagraphs(page: Page, api: TessBaseAPI): Sequence<String> {
        return sequence {
            val image = pixRead(page.file.toString())
            api.SetImage(image)
            api.SetVariable("preserve_interword_spaces", "1")

            val rects = detectParagraphs(page)
            for (rect in rects) {
                api.SetRectangle(rect.x(), rect.y(), rect.width(), rect.height())
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

    /**
     * Reads the [Page.file] as image, and returns the paragraphs
     * in an order according to a two-page layout.
     */
    private fun detectParagraphs(page: Page): List<Rect> {
        imread(page.file.toString()).use { original ->
            invertImage(original).use { inverted ->
                dilateContent(inverted).use { dilated ->
                    val contours = MatVector()
                    findContours(
                        dilated, contours,
                        RETR_EXTERNAL, CHAIN_APPROX_SIMPLE
                    )
                    return detectLayout(original, contours)
                }
            }
        }
    }

    private fun detectLayout(original: Mat, contours: MatVector): List<Rect> {
        val centerX = original.arrayWidth() / 2
        return contours.get()
            .map { boundingRect(it) }
            .sortedWith(Comparator<Rect> { left, right ->
                if (left.x() < centerX) {
                    if (right.x() < centerX) {
                        left.y() - right.y()
                    } else {
                        -1
                    }
                } else {
                    if (right.x() < centerX) {
                        1
                    } else {
                        left.y() - right.y()
                    }
                }
            })
    }

    private fun invertImage(src: Mat): Mat {
        return src.invertColors()
    }

    private fun dilateContent(src: Mat): Mat {
        val kernel = Mat.ones(10, 15, CV_8UC1).asMat()
        val dest = Mat()
        dilate(
            src, dest, kernel,
            Point(-1, -1), 3, BORDER_CONSTANT,
            morphologyDefaultBorderValue()
        )
        return dest
    }
}