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

import com.fasterxml.jackson.core.type.TypeReference
import java.io.File
import java.net.URI
import java.time.LocalDate
import java.time.format.DateTimeFormatter

/**
 * Constructs the directory structure for [Journal]
 */
internal fun directoryFor(baseDir: File, journal: Journal): File {
    val congress = journal.congress.number.toString()
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
    val format = DateTimeFormatter.ofPattern("yyyy-MM-dd")
    return """
    {
        "congress": ${congress.number},
        "chamber": "$chamber",
        "session": {
            "number": ${session.number},
            "type": "${session.type}"
        },
        "number": $number,
        "date": "${format.format(date)}",
        "document_uri": "$documentUri"
    }
    """.trimIndent()
}

internal fun Journal.Companion.fromJson(str: String): Journal {
    val map = OBJECT_MAPPER.readValue(str, object : TypeReference<Map<String, Any>>() {
    })
    val format = DateTimeFormatter.ofPattern("yyyy-MM-dd")
    val session = map["session"] as Map<String, Any>
    return Journal(
        congress = Congress(map["congress"] as Int),
        chamber = Chamber.valueOf(map["chamber"] as String),
        session = Session(
            number = session["number"] as Int,
            type = Session.Type.valueOf(session["type"] as String)
        ),
        number = map["number"] as Int,
        date = LocalDate.parse(map["date"] as String, format),
        documentUri = URI.create(map["document_uri"] as String)
    )
}