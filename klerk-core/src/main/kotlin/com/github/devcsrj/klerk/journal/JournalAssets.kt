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

import java.util.regex.Pattern

internal object JournalAssets {

    const val DOCUMENT = "document.pdf"
    const val PARSED_TEXT = "parsed.txt"

    fun documentPagePng(page: Int): String {
        return "page-" + "$page".padStart(3, '0') + ".png"
    }

    fun documentPagePngFilter() = Pattern.compile("page-\\d{3}\\.png").asPredicate()
}