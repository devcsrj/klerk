package com.github.devcsrj.klerk

import java.net.URI
import java.time.LocalDate

data class Journal(
    val congress: Congress,
    val session: Session,
    val number: Int,
    val date: LocalDate,
    val documentUri: URI
)