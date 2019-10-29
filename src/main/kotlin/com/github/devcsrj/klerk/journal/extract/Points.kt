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

import org.bytedeco.opencv.opencv_core.Point2f
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Computes the euclidian distance between this and [that] point.
 */
internal fun Point2f.distanceFrom(that: Point2f): Float {
    return sqrt(
        (that.x() - this.x()).pow(2) + (that.y() - this.y()).pow(2)
    )
}