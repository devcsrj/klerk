package com.github.devcsrj.klerk.senate

import com.github.devcsrj.klerk.Congress
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.gherkin.Feature
import java.time.LocalDate
import java.time.Month
import java.util.concurrent.TimeUnit
import kotlin.test.assertEquals
import kotlin.test.assertNull

object CommitteeReportItemReaderTest : Spek({

    val baseDir = "/senate/cr"
    val congress = Congress(17)

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
                                .getResourceAsStream("$baseDir/41.html")
                                .bufferedReader()
                                .readText()
                        )
                )
                reader.setCurrentItemCount(40)
            }

            When("read") {
                actual = reader.read()
            }

            Then("report is requested") {
                val rr = server.takeRequest(1, TimeUnit.SECONDS)!!
                val url = rr.requestUrl!!
                assertEquals("/lis/committee_rpt.aspx", url.encodedPath)
                assertEquals("17", url.queryParameter("congress"))
                assertEquals("41", url.queryParameter("q"))
            }

            Then("it should get the report") {
                val expected = CommitteeReport(
                    congress = congress,
                    number = 41,
                    title = "MENTAL HEALTH ACT OF 2017",
                    filingDate = LocalDate.of(2017, Month.FEBRUARY, 27),
                    document = server.url("/lisdata/2543921948!.pdf").toUri()
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
                                .getResourceAsStream("$baseDir/not-found.html")
                                .bufferedReader()
                                .readText()
                        )
                )
                reader.setCurrentItemCount(998)
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