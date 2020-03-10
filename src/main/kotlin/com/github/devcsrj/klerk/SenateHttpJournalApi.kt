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
package com.github.devcsrj.klerk

import okhttp3.*
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.logging.HttpLoggingInterceptor
import org.jsoup.nodes.Document
import org.jsoup.parser.Parser
import org.jsoup.select.Elements
import java.net.CookieManager
import java.net.CookiePolicy
import java.net.URI
import java.time.Duration
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util.regex.Pattern

/**
 * Note: This implementation is not thread-safe
 */
class SenateHttpJournalApi(private val url: URI) : JournalApi {

    private val baseUrl: HttpUrl = url.toHttpUrlOrNull()
        ?: throw IllegalArgumentException("Not a valid url: $url")
    private val client: OkHttpClient
    private val parser = Parser.htmlParser()

    companion object {
        private val DATE_FORMAT = DateTimeFormatter.ofPattern("MMMM d, yyyy")
        private val DATE_PATTERN =
            Pattern.compile(
                ".*Date: (" +
                        "(January" +
                        "|February" +
                        "|March" +
                        "|April" +
                        "|May" +
                        "|June" +
                        "|July" +
                        "|August" +
                        "|September" +
                        "|October" +
                        "|November" +
                        "|December) \\d{1,2}, \\d{4})"
            )
    }

    constructor() : this(URI.create("http://www.senate.gov.ph"))

    init {
        val loggingInterceptor = HttpLoggingInterceptor()
        loggingInterceptor.level = HttpLoggingInterceptor.Level.HEADERS

        // The site is built with ASP.NET, and relies on cookie to determine page state
        val cookieManager = CookieManager()
        cookieManager.setCookiePolicy(CookiePolicy.ACCEPT_ORIGINAL_SERVER)
        client = OkHttpClient.Builder()
            .cookieJar(JavaNetCookieJar(cookieManager))
            .addInterceptor(loggingInterceptor)
            .readTimeout(Duration.ofSeconds(30))
            .connectTimeout(Duration.ofSeconds(30))
            .build()
    }

    override fun fetch(congress: Congress, session: Session, offset: Int): Sequence<Journal> {
        if (offset > 0)
            TODO()
        return readJournals(congress, session)
    }

    private fun switchToCongressSession(congress: Congress, session: Session) {
        // ASP.NET pages are a nightmare to crawl, as they add additional
        // client state details in the pages.  So first we fetch the "landing page"
        val url = baseUrl.resolve("/lis/leg_sys.aspx")!!
            .newBuilder()
            .addQueryParameter("type", "journal")
            .addQueryParameter("congress", congress.number.toString())
            .build()
        val document = fetchDocument(url)

        val billType = session.let {
            it.number.toString() + it.type.name[0] // 1R, 2R, 3R, 1S, 2S, 3S
        }

        // Then we extract the required state variables
        val formBody = FormBody.Builder()
            .add("__EVENTTARGET", "dlBillType")
            .add("__EVENTARGUMENT", "")
            .add("__VIEWSTATE", document.selectFirst("#__VIEWSTATE").attr("value"))
            .add("__VIEWSTATEGENERATOR", document.selectFirst("#__VIEWSTATEGENERATOR").attr("value"))
            .add("__EVENTVALIDATION", document.selectFirst("#__EVENTVALIDATION").attr("value"))
            .add("dlBillType", billType)
            .build()

        // Finally, we resend the request with the state variables
        val request = Request.Builder()
            .url(url)
            .post(formBody)
            .build()
        client.newCall(request).execute().use {

        }
    }

    private fun readJournals(congress: Congress, session: Session): Sequence<Journal> {
        switchToCongressSession(congress, session)
        return readJournals(congress, session, 99)  // go to last page
    }

    private fun readJournals(congress: Congress, session: Session, page: Int): Sequence<Journal> {
        val url = baseUrl.resolve("/lis/leg_sys.aspx")!!
            .newBuilder()
            .addQueryParameter("type", "journal")
            .addQueryParameter("congress", congress.number.toString())
            .addQueryParameter("p", page.toString())
            .build()
        val document = fetchDocument(url)
        return readJournals(congress, session, document)
    }

    private fun readJournals(congress: Congress, session: Session, document: Document): Sequence<Journal> {
        val links = document.select("#lis_journal_table > div > * > a")
        val journals = readJournals(congress, session, links)
        val previous = document.selectFirst("#pnl_NavTop > div > div > a")
        val hasPrevious = previous.text() == "Previous"
        if (hasPrevious) {
            val href = previous.attr("href")
            val path = baseUrl.resolve(href)!!
            val p = path.queryParameter("p")?.toInt()
            return journals + readJournals(congress, session, p!!)
        }

        return journals
    }

    private fun readJournals(congress: Congress, session: Session, links: Elements): Sequence<Journal> {
        return sequence {
            for (a in links.reversed()) {
                val href = a.attr("href")
                val path = baseUrl.resolve("/lis/$href")!!
                yield(readJournal(congress, session, path))
            }
        }
    }

    private fun readJournal(congress: Congress, session: Session, url: HttpUrl): Journal {
        val document = fetchDocument(url)
        val body = document.body()
        val number = body
            .select("#content > div.lis_doctitle > p")
            .text()
            .substringAfter("Journal No. ")
            .trim()
            .toInt()
        val uri = body
            .select("#lis_download > ul > li > a")
            .attr("href")
        val date = body.select("#content").text().let {
            val matcher = DATE_PATTERN.matcher(it)
            require(matcher.find()) { "Could not find date from $url" }
            matcher.group(1)
        }

        return Journal(
            chamber = Chamber.SENATE,
            congress = congress,
            session = session,
            number = number,
            date = LocalDate.parse(
                date,
                DATE_FORMAT
            ),
            documentUri = baseUrl.resolve(uri)!!.toUri()
        )

    }

    private fun fetchDocument(url: HttpUrl): Document {
        val request = Request.Builder()
            .get().url(url)
            .build()

        return client.newCall(request).execute().use { response ->
            response.body.use { body ->
                val content = body!!.string()
                parser.parseInput(content, baseUrl.toString())
            }
        }
    }
}