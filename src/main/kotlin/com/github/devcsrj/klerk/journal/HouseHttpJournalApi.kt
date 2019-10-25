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
package com.github.devcsrj.klerk.journal

import com.github.devcsrj.klerk.Chamber
import com.github.devcsrj.klerk.Congress
import com.github.devcsrj.klerk.Journal
import com.github.devcsrj.klerk.Session
import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.OkHttpClient
import okhttp3.Request
import org.jsoup.nodes.Element
import org.jsoup.parser.Parser
import java.net.URI
import java.time.LocalDate
import java.time.format.DateTimeFormatter

class HouseHttpJournalApi(private val url: URI) : JournalApi {

    private val baseUrl: HttpUrl = url.toHttpUrlOrNull()
        ?: throw IllegalArgumentException("Not a valid url: $url")
    private val client = OkHttpClient()
    private val parser = Parser.htmlParser()

    companion object {
        private val DATE_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd")
    }

    constructor() : this(URI.create("http://www.congress.gov.ph"))

    override fun fetch(congress: Congress, session: Session, offset: Int): Iterator<Journal> {
        return readJournals(congress, session, offset).iterator()
    }

    private fun readJournals(congress: Congress, session: Session, offset: Int): Sequence<Journal> {
        val url = baseUrl.resolve("/legisdocs")!!
            .newBuilder()
            .addQueryParameter("v", "journals")
            .addQueryParameter("congress", congress.number.toString())
            .addQueryParameter("session", session.number.toString())
            .build()
        val request = Request.Builder()
            .get().url(url)
            .build()

        val document = client.newCall(request).execute().use { response ->
            response.body.use { body ->
                val reader = body!!.byteStream().bufferedReader()
                parser.parseInput(reader.readText(), url.toString())
            }
        }
        val trs = document.body().select("table > tbody > tr")
        return sequence {
            for (tr in trs.reversed()) {
                val journal = readJournal(congress, session, tr) ?: continue
                if (journal.number <= offset)
                    continue
                yield(journal)
            }
        }
    }

    private fun readJournal(congress: Congress, session: Session, tr: Element): Journal? {
        require(tr.tagName() == "tr") { "Expecting <tr>, but got: ${tr.tagName()}" }
        val number = tr.child(0)
            .text()
            .substringAfter("Journal No.")
            .trim()
            .toIntOrNull() ?: return null

        val date = LocalDate.parse(tr.child(1).text(), DATE_FORMAT)
        val href = tr.child(2).selectFirst("a").attr("href")
        return Journal(
            chamber = Chamber.HOUSE,
            congress = congress,
            session = session,
            number = number,
            date = date,
            documentUri = baseUrl.resolve(href)!!.toUri()
        )
    }
}