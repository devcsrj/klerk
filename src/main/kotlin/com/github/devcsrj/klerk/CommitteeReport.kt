package com.github.devcsrj.klerk

import java.net.URI
import java.time.LocalDate

data class CommitteeReport(

    val congress: Congress,
    val number: Int,
    val title: String,
    val filingDate: LocalDate?,
    val document: URI
)