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