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
package com.github.devcsrj.klerk.journal.collate

import com.github.devcsrj.klerk.Congress
import com.github.devcsrj.klerk.Journal
import com.github.devcsrj.klerk.Session
import com.github.devcsrj.klerk.journal.HouseHttpJournalApi
import com.github.devcsrj.klerk.journal.JournalApi
import com.github.devcsrj.klerk.journal.SenateHttpJournalApi
import org.apache.beam.sdk.transforms.DoFn
import org.slf4j.LoggerFactory

/**
 * Retrieves all [Session]s from each Congress of both the house and the senate
 */
internal class Fetch : DoFn<Congress, Journal>() {

    private val logger = LoggerFactory.getLogger(Fetch::class.java)

    private val sessions: Array<Session> = arrayOf(
        Session.regular(1),
        Session.regular(2),
        Session.regular(3)
    )

    @ProcessElement
    fun processElement(
        @Element congress: Congress,
        outputReceiver: OutputReceiver<Journal>
    ) {

        val houseApi: JournalApi =
            HouseHttpJournalApi()
        val senateApi: JournalApi =
            SenateHttpJournalApi()

        for (session in sessions) {
            val hs = houseApi.fetch(congress, session)
            for (journal in hs) {
                logger.info("📄️ $journal")
                outputReceiver.output(journal)
            }

            val ss = senateApi.fetch(congress, session)
            for (journal in ss) {
                logger.info("📄️ $journal")
                outputReceiver.output(journal)
            }
        }
    }
}