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
package com.github.devcsrj.klerk.journal

import com.fasterxml.jackson.core.type.TypeReference
import com.github.devcsrj.klerk.*
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