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

import java.nio.file.Path

/**
 * Constructs the directory structure for [Journal]
 */
internal fun directoryFor(baseDir: Path, journal: Journal): Path {
    val congress = journal.congress.number.ordinal()
    val session = journal.session.let {
        "${it.type.name.toLowerCase()}-${it.number}"
    }
    val chamber = journal.chamber.name
    return baseDir
        .resolve(congress)
        .resolve(chamber)
        .resolve(session)
}

internal fun Journal.asJson(): String {
    return OBJECT_MAPPER.writeValueAsString(this)
}

internal fun Journal.Companion.fromJson(str: String): Journal {
    return OBJECT_MAPPER.readValue(str, Journal::class.java)
}

/**
 * Returns the ordinal representation of this [Int]
 */
internal fun Int.ordinal(): String {
    val lastDigit = this % 10
    val lastTwoDigits = this % 100

    //Returns "th" on "teen" values with the last 2 digits being between 10 and 20
    if (lastTwoDigits in 10..20) {
        return this.toString() + "th"
    }

    //Returns appropriate suffix on non-"teen" values
    return when (lastDigit) {
        1 -> this.toString() + "st"
        2 -> this.toString() + "nd"
        3 -> this.toString() + "rd"
        else -> this.toString() + "th"
    }
}