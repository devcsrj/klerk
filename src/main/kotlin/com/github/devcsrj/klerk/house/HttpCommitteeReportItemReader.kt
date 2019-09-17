package com.github.devcsrj.klerk.house

import com.github.devcsrj.klerk.CommitteeReport
import com.github.devcsrj.klerk.Congress
import okhttp3.FormBody
import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.OkHttpClient
import okhttp3.Request
import org.jsoup.nodes.Document
import org.jsoup.nodes.Element
import org.jsoup.parser.Parser
import org.springframework.batch.item.support.AbstractItemCountingItemStreamItemReader
import org.springframework.util.ClassUtils
import java.net.URI
import java.time.LocalDate
import java.time.format.DateTimeFormatter

class HttpCommitteeReportItemReader(
    private val url: URI,
    private val congress: Congress
) : AbstractItemCountingItemStreamItemReader<CommitteeReport>() {

    private val baseUrl: HttpUrl
    private val client = OkHttpClient()
    private val parser = Parser.htmlParser()

    private lateinit var document: Document
    private var previousPanel: Element? = null

    companion object {
        private val FILE_DATE_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd")
    }

    init {
        setName(ClassUtils.getShortName(HttpCommitteeReportItemReader::class.java))
        baseUrl = url.toHttpUrlOrNull() ?: throw IllegalArgumentException("Not a valid url: $url")
    }

    override fun jumpToItem(itemIndex: Int) {
        while (true) {
            val headingPanel = previousPanel!!.nextElementSibling() ?: break
            val bodyPanel = headingPanel.nextElementSibling()
            previousPanel = bodyPanel

            if (readReportNumber(headingPanel) == itemIndex)
                break
        }
    }

    override fun doOpen() {
        val url = baseUrl.resolve("/committees")!!
            .newBuilder()
            .addQueryParameter("v", "reports")
            .addQueryParameter("congress", congress.number.toString())
            .build()
        val request = Request.Builder()
            .get().url(url)
            .build()

        document = client.newCall(request).execute().use { response ->
            response.body.use { body ->
                val reader = body!!.byteStream().bufferedReader()
                parser.parseInput(reader, url.toString())
            }
        }
        previousPanel = document.body()
            .selectFirst("div.panel-heading-custom")
    }

    override fun doRead(): CommitteeReport? {
        if (previousPanel == null)
            return null

        val headingPanel = previousPanel!!.nextElementSibling() ?: return null
        val bodyPanel = headingPanel.nextElementSibling()

        val reportNumber = readReportNumber(headingPanel)
        check(reportNumber == currentItemCount) {
            "Expecting report number $currentItemCount, but got $reportNumber"
        }

        val documentUri = readDocumentUri(headingPanel, reportNumber)
        val title = readTitle(bodyPanel, reportNumber)
        val filingDate = readFilingDate(bodyPanel)

        previousPanel = bodyPanel

        return CommitteeReport(
            congress = congress,
            number = reportNumber,
            title = title,
            filingDate = filingDate,
            document = documentUri
        )
    }

    private fun readFilingDate(panel: Element): LocalDate? {
        val a = panel.selectFirst("a[href^='#HistoryModal']")
        val form = FormBody.Builder()
            .add("rowid", a.attr("data-id"))
            .build()
        val request = Request.Builder()
            .post(form)
            .url(baseUrl.resolve("/committees/fetch_history.php")!!)
            .build()

        val history = client.newCall(request).execute().use { response ->
            response.body.use { body ->
                val reader = body!!.byteStream().bufferedReader()
                parser.parseInput(reader, url.toString())
            }
        }
        val text = history.select("td").asSequence()
            .map { it.text() }
            .filter { it.startsWith("DATE FILED") }
            .map { it.substringAfter("DATE FILED") }
            .map { it.substringAfterLast(":") }
            .map { it.trim() }
            .firstOrNull()
            ?: return null

        return LocalDate.parse(text, FILE_DATE_FORMAT)
    }

    private fun readTitle(panel: Element, reportNumber: Int): String {
        val p = panel.select("p").find { it.text().startsWith("TITLE:") }
        require(p != null) {
            "Could not find title of $congress, Committee Report $reportNumber"
        }
        return p.text().substringAfter("TITLE:").trim()
    }

    private fun readReportNumber(panel: Element): Int {
        return panel.select("span > strong").let { Integer.parseInt(it.text()) }
    }

    private fun readDocumentUri(panel: Element, reportNumber: Int): URI {
        val href = panel.select("span > a").attr("href")
        require(href != null) {
            "Could not extract pdf link of $congress, Committee Report $reportNumber"
        }
        return baseUrl.resolve(href)!!.toUri()
    }

    override fun doClose() {
        // noop
    }
}