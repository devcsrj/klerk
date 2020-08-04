package com.github.devcsrj.klerk.journal.collate

import com.github.devcsrj.klerk.Chamber
import com.github.devcsrj.klerk.Congress
import com.github.devcsrj.klerk.Session
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import java.net.URI
import java.time.LocalDate
import kotlin.test.assertEquals

object SenateHttpJournalApiFnTest : Spek({

    describe("api") {
        val api = SenateHttpJournalApi(URI.create("http://senate.gov.ph/"))

        it("should be able to fetch a journal") {
            val journals = api.fetch(Congress(17), Session.regular(3))
            val actual = journals.iterator().next()

            assertEquals(Congress(17), actual.congress)
            assertEquals(Session.regular(3), actual.session)
            assertEquals(Chamber.SENATE, actual.chamber)
            assertEquals(1, actual.number)
            assertEquals(LocalDate.of(2018, 7, 23), actual.date)
            assertEquals(URI.create("http://senate.gov.ph/lisdata/2821424504!.pdf"), actual.documentUri)
        }
    }
})