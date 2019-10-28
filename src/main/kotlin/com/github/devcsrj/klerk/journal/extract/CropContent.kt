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

import org.apache.beam.sdk.transforms.DoFn
import org.bytedeco.opencv.global.opencv_core.BORDER_CONSTANT
import org.bytedeco.opencv.global.opencv_core.CV_8UC1
import org.bytedeco.opencv.global.opencv_imgcodecs.imread
import org.bytedeco.opencv.global.opencv_imgcodecs.imwrite
import org.bytedeco.opencv.global.opencv_imgproc.*
import org.bytedeco.opencv.opencv_core.Mat
import org.bytedeco.opencv.opencv_core.MatVector
import org.bytedeco.opencv.opencv_core.Point
import org.bytedeco.opencv.opencv_core.Rect
import org.slf4j.LoggerFactory
import java.awt.Dimension
import java.io.File
import java.util.*
import kotlin.Comparator
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round


/**
 * Modifies image files by trimming out the border.
 *
 * The journal pages usually has a horizontal line at the top of the page.
 * Above it, are the page number, and the date. This function
 * removes that part.
 *
 * The new image is written as `${filename}-cropped.png`
 */
internal class CropContent : DoFn<File, File>() {

    private val logger = LoggerFactory.getLogger(CropContent::class.java)

    @ProcessElement
    fun processElement(
        @Element file: File,
        outputReceiver: OutputReceiver<File>
    ) {

        val name = file.nameWithoutExtension
        val dimension = readDimensions(file)
        val outputFile = file.parentFile.resolve("$name-cropped.png")
        if (outputFile.exists()) {
            outputReceiver.output(outputFile)
            return
        }

        logger.info("✂ $file")
        imread(file.toString()).use { src ->
            val region = invertImage(src).use { inverted ->
                dilateContent(inverted).use { dilated ->
                    // Mat().use { debug ->
                    //     src.copyTo(debug)
                    //     val f = file.parentFile.resolve("$name-debug.png")
                    //     dilated.drawContoursOn(debug)
                    //     imwrite(f.toString(), debug)
                    // }

                    findContent(dilated, dimension)
                }
            }
            if (region != null) {
                try {
                    val dest = Mat(src, region)
                    imwrite(outputFile.toString(), dest)
                    outputReceiver.output(outputFile)
                } catch (e: Exception) {
                    logger.error("⚠️ Failed to crop: $file", e)
                    outputReceiver.output(file) // the original
                }
            } else {
                outputReceiver.output(file) // the original
            }
        }

    }

    private fun readDimensions(file: File): Dimension {
        return Images.dimensionOf(file)
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

    private fun findContent(src: Mat, dimension: Dimension): Rect? {
        val contours = MatVector()
        findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

        val maxY = dimension.height * .15
        val minWidth = dimension.width / 2

        // Compares such that, the rectangle closest to the maxY is prioritized
        val closest = Comparator<Rect> { l, r -> ((maxY - l.y()) - (maxY - r.y())).toInt() }

        val matches = TreeSet(closest)
        // Look for the rectangle above the max Y
        for (contour in contours.get()) {
            val rect = boundingRect(contour)
            if (rect.y() > maxY)
                continue
            if (rect.width() < minWidth)
                continue
            matches.add(rect)
        }

        if (matches.isEmpty())
            return null // Does not exist

        val border = matches.first()
        val area = border.area()
        if (area <= 0)
            return null // Does not exist

        val maxArea = maxY * dimension.width
        if (border.height() > maxY || border.area() > maxArea)
            return null // Not expecting it to be too big

        val x = max(border.x() - 50, 0)
        val y = border.y() + border.height() - 10
        val width = min(border.width() + x + 50, dimension.width - x)
        val height = min(dimension.height - round(y * 1.5).toInt(), dimension.height - y)

        return Rect(x, y, width, height)
    }
}