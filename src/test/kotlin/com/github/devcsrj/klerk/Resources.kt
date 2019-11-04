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
package com.github.devcsrj.klerk

import java.io.File
import java.nio.file.Files

/**
 * Looks for the resource referred to by the [path], and
 * returns a copy as a temporary file.
 */
internal fun Class<*>.copyTempResource(path: String): File {
    val name = path.substringAfterLast('/')
    val prefix = name.substringBeforeLast('.')
    val suffix = name.substringAfterLast('.')
    val tmp = Files.createTempFile(prefix, suffix).toFile()
    tmp.deleteOnExit()
    tmp.outputStream().use { sink ->
        this.getResourceAsStream(path).use { src ->
            src.copyTo(sink)
        }
    }
    return tmp
}