package com.github.devcsrj.klerk.senate

import com.github.devcsrj.klerk.Congress
import com.github.devcsrj.klerk.Journal
import com.github.devcsrj.klerk.Session
import okhttp3.*
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import org.jsoup.nodes.Document
import org.jsoup.parser.Parser
import org.jsoup.select.Elements
import org.springframework.batch.item.ExecutionContext
import org.springframework.batch.item.support.AbstractItemStreamItemReader
import org.springframework.util.ClassUtils
import java.net.CookieManager
import java.net.CookiePolicy
import java.net.URI
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util.regex.Pattern

class HttpJournalItemReader(
    private val url: URI,
    private val congress: Congress
) : AbstractItemStreamItemReader<Journal>() {

    private val baseUrl: HttpUrl
    private val client: OkHttpClient
    private val parser = Parser.htmlParser()
    private val sessions = listOf(
        Session.regular(1),
        Session.regular(2),
        Session.regular(3),
        Session.special(1),
        Session.special(2),
        Session.special(3)
    )

    private var lastSession: Session? = null
    private var lastSessionIterator: Iterator<Session> = sessions.iterator()
    private var lastJournalNumber: Int = 0
    private var lastJournalIterator: Iterator<Journal> = object : Iterator<Journal> {
        override fun hasNext() = false
        override fun next(): Journal = throw NoSuchElementException()
    }

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

        private const val KEY_SESSION = "session"
        private const val KEY_JOURNAL_NUMBER = "journalNumber"
    }

    init {
        setName(ClassUtils.getShortName(HttpJournalItemReader::class.java))
        baseUrl = url.toHttpUrlOrNull() ?: throw IllegalArgumentException("Not a valid url: $url")

        // The site is built with ASP.NET, and relies on cookie to determine page state
        val cookieManager = CookieManager()
        cookieManager.setCookiePolicy(CookiePolicy.ACCEPT_ORIGINAL_SERVER)
        client = OkHttpClient.Builder()
            .cookieJar(JavaNetCookieJar(cookieManager))
            .build()
    }

    override fun open(executionContext: ExecutionContext) {
        lastSession = executionContext[KEY_SESSION] as Session?
        if (lastSession == null) {
            lastSession = Session.regular(1)
            lastSessionIterator = sessions.iterator()
        } else {
            val it = sessions.iterator()
            while (it.hasNext()) {
                val next = it.next()
                if (next == lastSession) {
                    break
                }
            }
            lastSessionIterator = it
        }
        lastJournalNumber = executionContext.getInt(KEY_JOURNAL_NUMBER, 0)
        lastJournalIterator = readJournals(lastSession!!).iterator()
    }

    override fun update(executionContext: ExecutionContext) {
        executionContext.put(KEY_SESSION, lastSession)
        executionContext.put(KEY_JOURNAL_NUMBER, lastJournalNumber)
    }

    override fun read(): Journal? {
        if (lastJournalIterator.hasNext()) {
            val next = lastJournalIterator.next()
            lastJournalNumber = next.number
            return next
        }

        if (!lastSessionIterator.hasNext()) {
            return null // we've reached the end
        }
        lastSession = lastSessionIterator.next()
        changeSessionTo(lastSession!!)

        lastJournalNumber = 0
        val journals = readJournals(lastSession!!).iterator()
        if (!journals.hasNext()) {
            return null // we've reached the end
        }
        lastJournalIterator = journals

        return read()
    }

    private fun changeSessionTo(session: Session) {
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
        client.newCall(request).execute().use { response ->
            check(response.code == 302) { "Expecting a redirect, but got: ${response.code}" }
        }
    }

    private fun readJournals(session: Session): Sequence<Journal> {
        return readJournals(session, 99) // go to last page
    }

    private fun readJournals(session: Session, page: Int): Sequence<Journal> {
        val url = baseUrl.resolve("/lis/leg_sys.aspx")!!
            .newBuilder()
            .addQueryParameter("type", "journal")
            .addQueryParameter("congress", congress.number.toString())
            .addQueryParameter("p", page.toString())
            .build()
        val document = fetchDocument(url)
        return readJournals(session, document)
    }

    private fun readJournals(session: Session, document: Document): Sequence<Journal> {
        val links = document.select("#lis_journal_table > div > * > a")
        val journals = readJournals(session, links)
        val previous = document.selectFirst("#pnl_NavTop > div > div > a")
        val hasPrevious = previous.text() == "Previous"
        if (hasPrevious) {
            val href = previous.attr("href")
            val path = baseUrl.resolve(href)!!
            val p = path.queryParameter("p")?.toInt()
            return journals + readJournals(session, p!!)
        }

        return journals
    }

    private fun readJournals(session: Session, links: Elements): Sequence<Journal> {
        return sequence {
            for (a in links.reversed()) {
                val href = a.attr("href")
                val path = baseUrl.resolve("/lis/$href")!!
                val q = path.queryParameter("q")?.toInt()
                if (q == null || q <= lastJournalNumber)
                    continue

                yield(readJournal(session, path))
            }
        }
    }

    private fun readJournal(session: Session, url: HttpUrl): Journal {
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
            congress = congress,
            session = session,
            number = number,
            date = LocalDate.parse(date, DATE_FORMAT),
            documentUri = baseUrl.resolve(uri)!!.toUri()
        )

    }

    private fun fetchDocument(url: HttpUrl): Document {
        val request = Request.Builder()
            .get().url(url)
            .build()

        return client.newCall(request).execute().use { response ->
            response.body.use { body ->
                val reader = body!!.byteStream().bufferedReader()
                parser.parseInput(reader, baseUrl.toString())
            }
        }
    }
}