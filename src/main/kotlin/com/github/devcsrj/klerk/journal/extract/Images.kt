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
    for ((i, contour) in contours.get().withIndex()) {
        val rect = boundingRect(contour)
        val label = "[$i] (${rect.x()}, ${rect.y()}), " +
                "${rect.width()}w x ${rect.height()}h "
        val color = Scalar.GREEN
        rectangle(output, rect, color)
        putText(
            output, label, Point(rect.x(), rect.y() + 10),
            FONT_HERSHEY_PLAIN, 1.0, Scalar.RED
        )
    }
}