package com.github.devcsrj.klerk.house

import com.github.devcsrj.klerk.CommitteeReport
import com.github.devcsrj.klerk.Congress
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.hamcrest.CoreMatchers.containsString
import org.hamcrest.MatcherAssert.assertThat
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.gherkin.Feature
import org.springframework.batch.item.ExecutionContext
import java.time.LocalDate
import java.util.concurrent.TimeUnit
import kotlin.test.assertEquals
import kotlin.test.assertNull

object CommitteeReportItemReaderTest : Spek({

    val baseDir = "/house/cr"
    val congress = Congress(18)

    Feature("Item reader") {
        lateinit var server: MockWebServer
        lateinit var reader: CommitteeReportItemReader

        beforeGroup {
            server = MockWebServer()
            reader = CommitteeReportItemReader(server.url("/").toUri(), congress)
        }

        afterGroup {
            server.shutdown()
        }

        Scenario("getting one item") {

            var actual: CommitteeReport? = null

            Given("one result") {
                server.enqueue(
                    MockResponse()
                        .setResponseCode(200)
                        .setBody(
                            javaClass
                                .getResourceAsStream("$baseDir/18th.html")
                                .bufferedReader()
                                .readText()
                        )
                )
                server.enqueue(
                    MockResponse()
                        .setResponseCode(200)
                        .setBody(
                            javaClass
                                .getResourceAsStream("$baseDir/18th-HB300-history.html")
                                .bufferedReader()
                                .readText()
                        )
                )
                reader.setCurrentItemCount(2)
                reader.open(ExecutionContext())
            }

            When("read") {
                actual = reader.read()
            }

            Then("report is requested") {
                val rr = server.takeRequest(1, TimeUnit.SECONDS)!!
                val url = rr.requestUrl!!
                assertEquals(rr.method, "GET")
                assertEquals("/committees", url.encodedPath)
                assertEquals("18", url.queryParameter("congress"))
                assertEquals("reports", url.queryParameter("v"))
            }

            Then("history is requested") {
                val rr = server.takeRequest(1, TimeUnit.SECONDS)!!
                val url = rr.requestUrl!!
                assertEquals(rr.method, "POST")
                assertEquals("/committees/fetch_history.php", url.encodedPath)
                assertThat(rr.body.readUtf8(), containsString("rowid=%23HB00300-18"))
            }

            Then("it should get the report") {
                val expected = CommitteeReport(
                    congress = congress,
                    number = 3,
                    title = "AN ACT AMENDING SECTIONS 4 AND 8 OF REPUBLIC ACT NO. 7042, " +
                            "AS AMENDED, OTHERWISE KNOWN AS THE 'FOREIGN INVESTMENT ACT OF 1991'",
                    filingDate = LocalDate.of(2019, 7, 1),
                    document = server.url("/legisdocs/first_18/CR00003.pdf").toUri()
                )
                assertEquals(expected, actual)
            }
        }

        Scenario("getting non-existent item") {

            var actual: CommitteeReport? = null

            Given("no result") {
                server.enqueue(
                    MockResponse()
                        .setResponseCode(200)
                        .setBody(
                            javaClass
                                .getResourceAsStream("$baseDir/18th.html")
                                .bufferedReader()
                                .readText()
                        )
                )
                reader.setCurrentItemCount(14)
                reader.open(ExecutionContext())
            }

            When("read") {
                actual = reader.read()
            }

            Then("it should return null") {
                assertNull(actual)
            }
        }
    }
})