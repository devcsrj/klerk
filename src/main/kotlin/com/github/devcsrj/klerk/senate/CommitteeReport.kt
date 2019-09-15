package com.github.devcsrj.klerk.senate

import com.github.devcsrj.klerk.Congress
import java.net.URI
import java.time.LocalDate

data class CommitteeReport(

    val congress: Congress,
    val number: Int,
    val title: String,
    val filingDate: LocalDate,
    val document: URI
)