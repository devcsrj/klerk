package com.github.devcsrj.klerk.legislator

import com.github.devcsrj.klerk.Legislator

interface LegislatorApi {

    fun fetch(): Set<Legislator>
}