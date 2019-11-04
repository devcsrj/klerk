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

import java.io.File
import java.io.Serializable

internal data class PageSlice(
    val page: Int,
    val index: Int,
    val file: File,
    val mat: ImageArray
) : Serializable, Comparable<PageSlice> {

    override fun compareTo(other: PageSlice): Int {
        val i = this.page - other.page
        return if (i != 0) i else this.index - other.index
    }

    override fun toString() = "Page $page, Section $index"
}