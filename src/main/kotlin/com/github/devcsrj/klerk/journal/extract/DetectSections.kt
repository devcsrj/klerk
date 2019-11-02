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
import org.bytedeco.opencv.global.opencv_core.BORDER_CONSTANT
import org.bytedeco.opencv.global.opencv_core.CV_8UC1
import org.bytedeco.opencv.global.opencv_imgcodecs.imread
import org.bytedeco.opencv.global.opencv_imgproc.*
import org.bytedeco.opencv.opencv_core.Mat
import org.bytedeco.opencv.opencv_core.MatVector
import org.bytedeco.opencv.opencv_core.Point
import org.bytedeco.opencv.opencv_core.Rect
import java.awt.Rectangle

/**
 * Loads a [Page.file] and slices the page into sections.
 */
internal class DetectSections : DoFn<
        KV<Journal, Iterable<@JvmWildcard Page>>,
        KV<Journal, PageSection>>() {

    @DoFn.ProcessElement
    fun processElement(context: ProcessContext) {

        val element = context.element()
        val journal = element.key!!
        for (page in element.value) {
            val sections = detectSections(page)
            for (section in sections) {
                context.output(KV.of(journal, section))
            }
        }
    }

    /**
     * Reads the [Page.file] as image, and returns the paragraphs
     * in an order according to a two-page layout.
     */
    private fun detectSections(page: Page): List<PageSection> {
        imread(page.file.toString()).use { original ->
            invertImage(original).use { inverted ->
                dilateContent(inverted).use { dilated ->
                    val contours = MatVector()
                    findContours(
                        dilated, contours,
                        RETR_EXTERNAL, CHAIN_APPROX_SIMPLE
                    )
                    return detectRectangles(original, contours)
                        .map { Rectangle(it.x(), it.y(), it.width(), it.height()) }
                        .mapIndexed { i, rect -> PageSection(page.number, i, rect, page.file) }
                }
            }
        }
    }

    private fun detectRectangles(original: Mat, contours: MatVector): List<Rect> {
        val rects = contours.get().map { boundingRect(it) }
        var leftmost = original.arrayWidth()
        var rightmost = 0
        for (rect in rects) {
            if (rect.x() < leftmost)
                leftmost = rect.x()
            if (rect.x() > rightmost)
                rightmost = rect.x()
        }
        val centerX = (leftmost + rightmost) / 2

        return rects.sortedWith(Comparator<Rect> { left, right ->
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
        val kernel = Mat.ones(10, 25, CV_8UC1).asMat()
        val dest = Mat()
        dilate(
            src, dest, kernel,
            Point(-1, -1), 3, BORDER_CONSTANT,
            morphologyDefaultBorderValue()
        )
        return dest
    }
}