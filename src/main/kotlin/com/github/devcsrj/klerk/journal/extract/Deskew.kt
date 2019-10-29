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
import org.bytedeco.opencv.global.opencv_core.*
import org.bytedeco.opencv.global.opencv_imgcodecs.imread
import org.bytedeco.opencv.global.opencv_imgcodecs.imwrite
import org.bytedeco.opencv.global.opencv_imgproc.*
import org.bytedeco.opencv.opencv_core.*
import org.slf4j.LoggerFactory
import kotlin.math.roundToInt

/**
 * Deskews image files.
 *
 * Since the PDFs are usually a result from a scanner, the ending
 * file sometimes is skewed. This step ensures that the content
 * is upright prior OCR.
 */
internal class Deskew : DoFn<KV<Journal, Page>, KV<Journal, Page>>() {

    private val logger = LoggerFactory.getLogger(Deskew::class.java)

    @ProcessElement
    fun processElement(context: ProcessContext) {

        val element = context.element()
        val journal = element.key!!
        val page = element.value

        logger.info("📏 Journal ${journal.number} - $page")
        val deskewedPage = deskew(journal, page)
        context.output(KV.of(journal, deskewedPage))
    }

    private fun deskew(journal: Journal, page: Page): Page {
        val file = page.file
        val name = "journal-${journal.number}-p${page.number}-deskewed.png"
        val outputFile = file.parentFile.resolve(name)
        val deskewedPage = Page(
            number = page.number,
            file = outputFile
        )
        if (outputFile.exists())
            return deskewedPage

        val deskewed = imread(file.toString()).use { src ->
            invertImage(src).use { inverted ->
                dilateContent(inverted).use { dilated ->
                    // Mat().use { debug ->
                    //     src.copyTo(debug)
                    //     val f = file.parentFile.resolve("$name-debug.png")
                    //     dilated.drawContoursOn(debug)
                    //     imwrite(f.toString(), debug)
                    // }

                    deskew(src, dilated)
                }
            }
        }
        return try {
            imwrite(outputFile.toString(), deskewed)
            deskewedPage
        } catch (e: Exception) {
            logger.error("⚠️ Failed to deskew: $file", e)
            page
        }
    }

    private fun deskew(src: Mat, dilated: Mat): Mat {
        val contours = MatVector()
        findContours(dilated, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

        val centerX = src.cols() / 2F
        val centerY = src.rows() / 2F
        val center = Point2f(centerX, centerY)

        // Select the rectagle closest to the center
        val closest = Comparator<RotatedRect> { left, right ->
            (left.center().distanceFrom(center) - right.center().distanceFrom(center)).roundToInt()
        }

        // The selected base rectangle should not be the entire page
        val reduction = .10
        val maxArea = src.rows().let { it - (it * reduction) } *
                src.cols().let { it - (it * reduction) }

        val centerMost = contours.get()
            .map { minAreaRect(it) }
            .filter { it.boundingRect().area() < maxArea }
            .sortedWith(closest)
            .firstOrNull() ?: return src

        return deskew(src, centerMost)
    }

    private fun deskew(src: Mat, rect: RotatedRect): Mat {
        // https://stackoverflow.com/questions/15956124
        val angle = rect.angle().let {
            if (it < -45) {
                (90.0 + it)
            } else {
                it.toDouble()
            }
        }
        val size = src.size()
        val center = size.let {
            Point2f(it.width() / 2.0F, it.height() / 2.0F)
        }
        val rotation = getRotationMatrix2D(center, angle, 1.0)

        val result = Mat(src.rows(), src.cols(), CV_8UC1, Scalar.WHITE)
        warpAffine(
            src, result, rotation,
            size, CV_INTER_CUBIC, BORDER_REPLICATE,
            morphologyDefaultBorderValue()
        )

        return result
    }


    private fun invertImage(src: Mat): Mat {
        return src.invertColors()
    }

    private fun dilateContent(src: Mat): Mat {
        val kernel = Mat.ones(10, 15, CV_8UC1).asMat()
//        val kernel = Mat.ones(100, 60, CV_8UC1).asMat()
        val dest = Mat()
        dilate(
            src, dest, kernel,
            Point(-1, -1), 3, BORDER_CONSTANT,
            morphologyDefaultBorderValue()
        )
        return dest
    }
}