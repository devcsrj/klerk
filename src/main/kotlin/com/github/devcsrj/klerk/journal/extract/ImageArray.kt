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

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.opencv.global.opencv_core.CV_8UC
import org.bytedeco.opencv.opencv_core.Mat
import java.io.Serializable

internal data class ImageArray(
    val rows: Int,
    val cols: Int,
    val channels: Int,
    val bytes: ByteArray
) : Serializable {

    companion object {

        fun create(mat: Mat): ImageArray {
            val b = ByteArray(mat.channels() * mat.cols() * mat.rows())
            mat.data().get(b)
            return ImageArray(mat.rows(), mat.cols(), mat.channels(), b)
        }
    }

    fun toMat(): Mat {
        return Mat(rows, cols, CV_8UC(channels), BytePointer(*bytes))
    }

    override fun toString() = "$rows x $cols, $channels"

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as ImageArray

        if (rows != other.rows) return false
        if (cols != other.cols) return false
        if (channels != other.channels) return false
        if (!bytes.contentEquals(other.bytes)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = rows
        result = 31 * result + cols
        result = 31 * result + channels
        result = 31 * result + bytes.contentHashCode()
        return result
    }
}