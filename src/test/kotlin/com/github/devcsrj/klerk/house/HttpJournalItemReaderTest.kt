package com.github.devcsrj.klerk.house

import com.github.devcsrj.klerk.Congress
import com.github.devcsrj.klerk.Journal
import com.github.devcsrj.klerk.Session
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.gherkin.Feature
import org.springframework.batch.item.ExecutionContext
import java.time.LocalDate
import java.time.Month
import java.util.concurrent.TimeUnit
import kotlin.test.assertEquals

object HttpJournalItemReaderTest : Spek({

    val baseDir = "/house/journal"
    val congress = Congress(17)

    Feature("Item reader") {
        lateinit var server: MockWebServer
        lateinit var readerHttp: HttpJournalItemReader

        beforeGroup {
            server = MockWebServer()
            readerHttp = HttpJournalItemReader(server.url("/").toUri(), congress)
        }

        afterGroup {
            server.shutdown()
        }

        Scenario("getting one item") {

            var actual: Journal? = null

            Given("one page") {
                server.enqueue(
                    MockResponse()
                        .setResponseCode(200)
                        .setBody(
                            javaClass
                                .getResourceAsStream("$baseDir/17th-session1.html")
                                .bufferedReader()
                                .readText()
                        )
                )
                readerHttp.open(ExecutionContext())
            }

            When("read") {
                actual = readerHttp.read()
            }

            Then("journal is requested") {
                val rr = server.takeRequest(1, TimeUnit.SECONDS)!!
                val url = rr.requestUrl!!
                assertEquals(rr.method, "GET")
                assertEquals("/legisdocs", url.encodedPath)
                assertEquals("17", url.queryParameter("congress"))
                assertEquals("1", url.queryParameter("session"))
                assertEquals("journals", url.queryParameter("v"))
            }

            Then("it should get the journal") {
                val expected = Journal(
                    congress = congress,
                    session = Session.regular(1),
                    number = 1,
                    date = LocalDate.of(2016, Month.JULY, 25),
                    documentUri = server.url("/legisdocs/journals_17/J1-1RS-20160725.pdf").toUri()
                )
                assertEquals(expected, actual)
            }
        }

        Scenario("read all") {

            var actual = 0

            Given("all pages") {
                for (i in 1..4) {
                    val path = "$baseDir/17th-session$i.html"
                    server.enqueue(
                        MockResponse()
                            .setResponseCode(200)
                            .setBody(
                                javaClass
                                    .getResourceAsStream(path)
                                    .bufferedReader()
                                    .readText()
                            )
                    )
                }
                readerHttp.open(ExecutionContext())
            }

            When("read all") {
                while (true) {
                    readerHttp.read() ?: break
                    actual++
                }
            }

            Then("total is correct") {
                assertEquals(54 + 86 + 97, actual)
            }
        }

    }
})