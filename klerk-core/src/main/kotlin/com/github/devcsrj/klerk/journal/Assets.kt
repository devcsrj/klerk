/**
 * Copyright [2020] [Reijhanniel Jearl Campos]
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
package com.github.devcsrj.klerk.journal

import java.io.OutputStream
import java.nio.file.Path
import java.util.function.Predicate

interface Assets {

    companion object {

        val EMPTY = object : Assets {
            override fun sink(name: String) = throw UnsupportedOperationException()
            override fun file(name: String) = throw UnsupportedOperationException()
            override fun list(filenameFilter: Predicate<String>) = emptyList<Path>()
        }
    }

    fun sink(name: String): OutputStream
    fun file(name: String): Path
    fun list(filenameFilter: Predicate<String>): List<Path>
}