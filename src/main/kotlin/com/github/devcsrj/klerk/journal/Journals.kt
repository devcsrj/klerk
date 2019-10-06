package com.github.devcsrj.klerk.journal

import com.github.devcsrj.klerk.Journal
import java.io.File

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