package com.github.devcsrj.klerk.house

import com.github.devcsrj.klerk.Congress
import com.github.devcsrj.klerk.Journal
import com.github.devcsrj.klerk.Session
import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.OkHttpClient
import okhttp3.Request
import org.jsoup.nodes.Element
import org.jsoup.parser.Parser
import org.springframework.batch.item.ExecutionContext
import org.springframework.batch.item.support.AbstractItemStreamItemReader
import org.springframework.util.ClassUtils
import java.net.URI
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util.*
import kotlin.NoSuchElementException

class HttpJournalItemReader(
    private val url: URI,
    private val congress: Congress
) : AbstractItemStreamItemReader<Journal>() {

    private val baseUrl: HttpUrl
    private val client = OkHttpClient()
    private val parser = Parser.htmlParser()

    private var lastSession: Session? = null
    private var lastJournalNumber: Int = 0
    private var lastIterator: Iterator<Journal> = object : Iterator<Journal> {
        override fun hasNext() = false
        override fun next(): Journal = throw NoSuchElementException()
    }

    companion object {
        private val DATE_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd")
        private const val KEY_SESSION = "session"
        private const val KEY_JOURNAL_NUMBER = "journalNumber"
    }

    init {
        setName(ClassUtils.getShortName(HttpJournalItemReader::class.java))
        baseUrl = url.toHttpUrlOrNull() ?: throw IllegalArgumentException("Not a valid url: $url")
    }

    override fun open(executionContext: ExecutionContext) {
        lastSession = executionContext[KEY_SESSION] as Session?
        if (lastSession == null)
            lastSession = Session.regular(1)
        lastJournalNumber = executionContext.getInt(KEY_JOURNAL_NUMBER, 0)
        lastIterator = readJournals(lastSession!!).iterator()
        while (lastJournalNumber > 0 && lastIterator.hasNext()) {
            val next = lastIterator.next()
            if (next.number >= lastJournalNumber)
                break
        }
    }

    override fun update(executionContext: ExecutionContext) {
        executionContext.put(KEY_SESSION, lastSession)
        executionContext.put(KEY_JOURNAL_NUMBER, lastJournalNumber)
    }

    override fun read(): Journal? {
        if (lastIterator.hasNext()) {
            val next = lastIterator.next()
            lastJournalNumber = next.number
            return next
        }

        lastSession = Session.regular(lastSession!!.number + 1)
        lastJournalNumber = 0
        val journals = readJournals(lastSession!!)
        if (journals.isEmpty()) {
            return null // we've reached the end
        }
        lastIterator = journals.iterator()

        return read()
    }

    private fun readJournals(session: Session): Collection<Journal> {
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
                parser.parseInput(reader, url.toString())
            }
        }
        val trs = document.body().select("table > tbody > tr")
        if (trs.isEmpty())
            return emptyList()

        val list = LinkedList<Journal>()
        for (tr in trs) {
            val journal = readJournal(session, tr) ?: continue
            list.addFirst(journal)
        }
        return list
    }

    private fun readJournal(session: Session, tr: Element): Journal? {
        require(tr.tagName() == "tr") { "Expecting <tr>, but got: ${tr.tagName()}" }
        val number = tr.child(0)
            .text()
            .substringAfter("Journal No.")
            .trim()
            .toIntOrNull() ?: return null

        val date = LocalDate.parse(tr.child(1).text(), DATE_FORMAT)
        val href = tr.child(2).selectFirst("a").attr("href")
        return Journal(
            congress = congress,
            session = session,
            number = number,
            date = date,
            documentUri = baseUrl.resolve(href)!!.toUri()
        )
    }
}