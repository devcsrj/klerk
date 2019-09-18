package com.github.devcsrj.klerk.journal

import com.github.devcsrj.klerk.Congress
import com.github.devcsrj.klerk.Journal
import com.github.devcsrj.klerk.Session

interface JournalApi {

    fun fetch(congress: Congress, session: Session, offset: Int = 0): Iterator<Journal>
}