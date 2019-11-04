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

import org.bytedeco.leptonica.PIX
import org.bytedeco.leptonica.global.lept
import org.bytedeco.opencv.global.opencv_imgproc.*
import org.bytedeco.opencv.opencv_core.Mat
import org.bytedeco.opencv.opencv_core.MatVector
import org.bytedeco.opencv.opencv_core.Point
import org.bytedeco.opencv.opencv_core.Scalar
import java.awt.Dimension
import java.io.File
import java.io.IOException
import javax.imageio.ImageIO
import javax.imageio.stream.FileImageInputStream

object Images {

    /**
     * Retrieves the dimension for the provided image file.
     */
    fun dimensionOf(imgFile: File): Dimension {
        val pos = imgFile.name.lastIndexOf(".")
        if (pos == -1)
            throw IOException("No extension for file: " + imgFile.absolutePath)
        val suffix = imgFile.name.substring(pos + 1)
        val iter = ImageIO.getImageReadersBySuffix(suffix)
        while (iter.hasNext()) {
            val reader = iter.next()
            try {
                val stream = FileImageInputStream(imgFile)
                reader.input = stream
                val width = reader.getWidth(reader.minIndex)
                val height = reader.getHeight(reader.minIndex)
                return Dimension(width, height)
            } catch (e: IOException) {
                // ignore
            } finally {
                reader.dispose()
            }
        }

        throw IOException("Not a known image file: " + imgFile.absolutePath)
    }
}

/**
 * Draws rectangles on the provided [output]
 */
internal fun Mat.drawContoursOn(output: Mat) {
    val contours = MatVector()
    findContours(this, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    contours.drawOn(output)
}

internal fun MatVector.drawOn(output: Mat) {
    for ((i, contour) in this.get().withIndex()) {
        val minRect = minAreaRect(contour)
        val rect = minRect.boundingRect()
        val label = "[$i] (${rect.x()}, ${rect.y()}), " +
                "${rect.width()}w x ${rect.height()}h " +
                "- ${minRect.angle()} degrees"
        val color = Scalar.GREEN
        rectangle(output, rect, color)
        putText(
            output, label, Point(rect.x(), rect.y() + 10),
            FONT_HERSHEY_PLAIN, 1.0, Scalar.RED
        )
    }
}

/**
 * Creates a matrix that is a color inverted version
 * of this matrix.
 */
internal fun Mat.invertColors(): Mat {
    return Mat().use { grey ->
        cvtColor(this, grey, COLOR_BGR2GRAY)
        val dest = Mat()
        threshold(grey, dest, 0.0, 255.0, THRESH_BINARY_INV + THRESH_OTSU)
        dest
    }
}

/**
 * Converts this [Mat] into a leptonica [PIX] type
 */
internal fun Mat.toPix(): PIX {
    val rows = this.rows()
    val cols = this.cols()
    val pix = lept.pixCreate(cols, rows, 8)
    for (y in 0..rows) {
        for (x in 0..cols) {
            val ptr = this.ptr(y, x)
            lept.pixSetPixel(pix, x, y, ptr.int)
        }
    }
    return pix
}
