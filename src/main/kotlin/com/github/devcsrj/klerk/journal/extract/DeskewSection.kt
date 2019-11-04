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
import org.bytedeco.opencv.global.opencv_imgproc.*
import org.bytedeco.opencv.opencv_core.*
import org.slf4j.LoggerFactory

/**
 * Deskews image files.
 *
 * Since the PDFs are usually a result from a scanner, the ending
 * file sometimes is skewed. This step ensures that the content
 * is upright prior OCR.
 */
internal class DeskewSection : DoFn<
        KV<Journal, PageSlice>,
        KV<Journal, PageSlice>>() {

    private val logger = LoggerFactory.getLogger(DeskewSection::class.java)

    @ProcessElement
    fun processElement(context: ProcessContext) {

        val element = context.element()
        val journal = element.key!!
        val slice = element.value

        logger.info("📏 $slice ($journal)")
        val deskewed = deskew(slice.mat.toMat())
        val newSlice = PageSlice(
            page = slice.page,
            index = slice.index,
            file = slice.file,
            mat = ImageArray.create(deskewed)
        )
        context.output(KV.of(journal, newSlice))
    }

    private fun deskew(src: Mat): Mat {
        invertImage(src).use { inverted ->
            dilateContent(inverted).use { dilated ->
                val contours = MatVector()
                findContours(dilated, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
                return deskew(src, contours)
            }
        }
    }

    private fun deskew(src: Mat, contours: MatVector): Mat {
        val contour = contours.get().first() ?: return Mat()
        val rect = minAreaRect(contour)
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

    private fun invertImage(src: Mat): Mat {
        return src.invertColors()
    }

}