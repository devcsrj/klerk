package com.github.devcsrj.klerk.senate

import com.github.devcsrj.klerk.Congress
import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.OkHttpClient
import okhttp3.Request
import org.jsoup.nodes.Document
import org.jsoup.parser.Parser
import org.springframework.batch.item.support.AbstractItemCountingItemStreamItemReader
import org.springframework.util.ClassUtils
import java.net.URI
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util.regex.Pattern

class CommitteeReportItemReader(
    private val url: URI,
    private val congress: Congress
) : AbstractItemCountingItemStreamItemReader<CommitteeReport>() {

    private val baseUrl: HttpUrl
    private val client = OkHttpClient()
    private val parser = Parser.htmlParser()

    companion object {
        private val FILE_DATE_PATTERN =
            Pattern.compile("Filed on ((January|February|March|April|May|June|July|August|September|October|November|December) \\d{1,2}, \\d{4})")
        private val FILE_DATE_FORMAT =
            DateTimeFormatter.ofPattern("MMMM d, yyyy")
    }

    init {
        setName(ClassUtils.getShortName(CommitteeReportItemReader::class.java))
        baseUrl = url.toHttpUrlOrNull() ?: throw IllegalArgumentException("Not a valid url: $url")
    }

    override fun jumpToItem(itemIndex: Int) {
        // noop
    }

    override fun doOpen() {
        // noop
    }

    override fun doRead(): CommitteeReport? {
        val reportNumber = currentItemCount
        val request = requestFor(reportNumber)
        return client.newCall(request).execute().use { response ->
            val doc = response.body.use { body ->
                val reader = body!!.byteStream().bufferedReader()
                parser.parseInput(reader, url.toString())
            }
            reportFrom(reportNumber, doc)
        }
    }

    override fun doClose() {
        // noop
    }

    private fun reportFrom(
        reportNumber: Int,
        document: Document
    ): CommitteeReport? {

        val body = document.body()
        val content = body.select("#content")
        val contentText = content.text()
        if (contentText.contains("Not found")) {
            return null
        }

        val title = content.select("div.lis_doctitle > p").text()
        val filingDate = contentText.let {
            val matcher = FILE_DATE_PATTERN.matcher(it)
            require(matcher.find()) {
                "Could not extract filing date of $congress, Committee Report $reportNumber"
            }
            val date = matcher.group(1)
            LocalDate.parse(date, FILE_DATE_FORMAT)
        }
        val href = content.select("#lis_download > ul > li > a")
            .find { a -> a.text() == "CR-$reportNumber" }?.attr("href")
        require(href != null) { "Could not extract pdf link of $congress, Committee Report $reportNumber" }

        return CommitteeReport(
            congress = congress,
            number = reportNumber,
            title = title,
            filingDate = filingDate,
            document = baseUrl.resolve(href)!!.toUri()
        )
    }

    private fun requestFor(reportNumber: Int): Request {
        val url = baseUrl.resolve("/lis/committee_rpt.aspx")!!
            .newBuilder()
            .addQueryParameter("congress", congress.number.toString())
            .addQueryParameter("q", reportNumber.toString())
            .build()
        return Request.Builder()
            .get()
            .url(url)
            .build()
    }

}