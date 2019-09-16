package com.github.devcsrj.klerk.senate

import com.github.devcsrj.klerk.Congress
import com.github.devcsrj.klerk.Journal
import com.github.devcsrj.klerk.Session
import com.github.devcsrj.klerk.enqueue
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.hamcrest.CoreMatchers.containsString
import org.hamcrest.MatcherAssert.assertThat
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.gherkin.Feature
import org.springframework.batch.item.ExecutionContext
import java.time.LocalDate
import java.time.Month
import java.util.concurrent.TimeUnit
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

object JournalItemReaderTest : Spek({

    val baseDir = "/senate/journal"
    val congress = Congress(17)

    Feature("Item Reader") {
        lateinit var server: MockWebServer
        lateinit var reader: JournalItemReader

        beforeGroup {
            server = MockWebServer()
            reader = JournalItemReader(server.url("/").toUri(), congress)
        }

        afterGroup {
            server.shutdown()
        }

        Scenario("getting one item") {
            var actual: Journal? = null

            Given("one page") {
                server.enqueue(200, "$baseDir/17th-regular_session1-p4.html")
                server.enqueue(200, "$baseDir/17th-regular_session1-p3.html")
                server.enqueue(200, "$baseDir/17th-regular_session1-p2.html")
                server.enqueue(200, "$baseDir/17th-regular_session1-p1.html")
                server.enqueue(200, "$baseDir/17th-regular_session1-journal1.html")
                reader.open(ExecutionContext())
            }

            When("read") {
                actual = reader.read()
            }

            Then("journals are fetched") {
                val rr = server.takeRequest(1, TimeUnit.SECONDS)!!
                val url = rr.requestUrl!!
                assertEquals(rr.method, "GET")
                assertEquals("/lis/leg_sys.aspx", url.encodedPath)
                assertEquals("17", url.queryParameter("congress"))
                assertEquals("99", url.queryParameter("p"))
                assertEquals("journal", url.queryParameter("type"))

                for (i in 3 downTo 1) {
                    server.takeRequest(1, TimeUnit.SECONDS)
                }
            }

            Then("journal is requested") {
                val rr = server.takeRequest(1, TimeUnit.SECONDS)!!
                val url = rr.requestUrl!!
                assertEquals(rr.method, "GET")
                assertEquals("/lis/journal.aspx", url.encodedPath)
                assertEquals("17", url.queryParameter("congress"))
                assertEquals("1R", url.queryParameter("session"))
                assertEquals("1", url.queryParameter("q"))
            }

            Then("it should get the journal") {
                val expected = Journal(
                    congress = congress,
                    session = Session.regular(1),
                    number = 1,
                    date = LocalDate.of(2016, Month.JULY, 25),
                    documentUri = server.url("/lisdata/2376420165!.pdf").toUri()
                )
                assertEquals(expected, actual)
            }
        }

        Scenario("next page") {
            var actual: Journal? = null

            Given("pages") {
                server.enqueue(200, "$baseDir/17th-regular_session1-p4.html")
                server.enqueue(200, "$baseDir/17th-regular_session1-p3.html")
                server.enqueue(200, "$baseDir/17th-regular_session1-p2.html")
                server.enqueue(200, "$baseDir/17th-regular_session1-p1.html")
                server.enqueue(200, "$baseDir/17th-regular_session1-journal1.html") // doesn't matter

                val ctx = ExecutionContext()
                ctx.putInt("journalNumber", 22)
                reader.open(ctx)
            }

            When("read") {
                actual = reader.read()
            }

            Then("previous pages are requested") {
                var rr = server.takeRequest(1, TimeUnit.SECONDS)!!
                var url = rr.requestUrl!!
                assertEquals(rr.method, "GET")
                assertEquals("/lis/leg_sys.aspx", url.encodedPath)
                assertEquals("17", url.queryParameter("congress"))
                assertEquals("99", url.queryParameter("p"))
                assertEquals("journal", url.queryParameter("type"))

                for (i in 3 downTo 1) {
                    rr = server.takeRequest(1, TimeUnit.SECONDS)!!
                    url = rr.requestUrl!!
                    assertEquals(rr.method, "GET")
                    assertEquals("/lis/leg_sys.aspx", url.encodedPath)
                    assertEquals("17", url.queryParameter("congress"))
                    assertEquals(i.toString(), url.queryParameter("p"))
                    assertEquals("journal", url.queryParameter("type"))
                }
            }

            Then("journal is requested") {
                val rr = server.takeRequest(1, TimeUnit.SECONDS)!!
                val url = rr.requestUrl!!
                assertEquals(rr.method, "GET")
                assertEquals("/lis/journal.aspx", url.encodedPath)
                assertEquals("17", url.queryParameter("congress"))
                assertEquals("1R", url.queryParameter("session"))
                assertEquals("23", url.queryParameter("q"))
            }

            Then("journal is not null") {
                assertNotNull(actual)
            }
        }

        Scenario("Next session") {
            var actual: Journal? = null

            Given("pages") {
                server.enqueue(200, "$baseDir/17th-regular_session1-p4.html")
                server.enqueue(200, "$baseDir/17th-regular_session1-p3.html")
                server.enqueue(200, "$baseDir/17th-regular_session1-p2.html")
                server.enqueue(200, "$baseDir/17th-regular_session1-p1.html")
                server.enqueue(200, "$baseDir/17th-regular_session1-p4.html") // takes the "landing page"
                server.enqueue(MockResponse().setResponseCode(302)) // requests a session change
                server.enqueue(200, "$baseDir/17th-regular_session2-p1.html") // skip the rest of the pages
                server.enqueue(200, "$baseDir/17th-regular_session1-journal1.html") // doesn't matter

                val ctx = ExecutionContext()
                ctx.put("session", Session.regular(1))
                ctx.putInt("journalNumber", 89)
                reader.open(ctx)
            }

            When("read") {
                actual = reader.read()
            }

            Then("selected regular session is changed") {
                for (i in 5 downTo 1) {
                    server.takeRequest(1, TimeUnit.SECONDS)!!
                }

                val rr = server.takeRequest(1, TimeUnit.SECONDS)!!
                val body = rr.body.readUtf8()
                assertThat(body, containsString("__EVENTTARGET=dlBillType"))
                assertThat(body, containsString("__EVENTARGUMENT="))
                assertThat(
                    body, containsString(
                        "__VIEWSTATE=%2FwEPDwULLTIwNjQzNjgxMjYPZBYCAgEPZBYMZ" +
                                "g8PFgIeB1Zpc2libGVoZBYCAgEPEGRkFgFmZAIBDw8WA" +
                                "h8AaGQWAgIBDxBkZBYBZmQCAg8PFgIfAGhkFgICAQ8QZ" +
                                "GQWAWZkAgMPZBYGAgEPEGQPFgZmAgECAgIDAgQCBRYGE" +
                                "AUVRmlyc3QgUmVndWxhciBTZXNzaW9uBQIxUmcQBRZTZ" +
                                "WNvbmQgUmVndWxhciBTZXNzaW9uBQIyUmcQBRVUaGlyZ" +
                                "CBSZWd1bGFyIFNlc3Npb24FAjNSZxAFFUZpcnN0IFNwZ" +
                                "WNpYWwgU2Vzc2lvbgUCMVNnEAUWU2Vjb25kIFNwZWNpY" +
                                "WwgU2Vzc2lvbgUCMlNnEAUVVGhpcmQgU3BlY2lhbCBTZ" +
                                "XNzaW9uBQIzU2cWAWZkAgMPDxYCHwBoZGQCBA8PFgIfA" +
                                "GhkZAIED2QWDGYPDxYCHgRUZXh0BQYmbmJzcDtkZAIBD" +
                                "w8WBB8BBQREYXRlHwBoZGQCAg8PFgIfAGhkZAIDDw8WA" +
                                "h8AaGRkAgQPDxYCHwBoZGQCBQ8PFgIfAGhkZAIFDw8WA" +
                                "h8AaGQWDGYPDxYCHwEFBiZuYnNwO2RkAgEPDxYEHwEFB" +
                                "ERhdGUfAGhkZAICDw8WAh8AaGRkAgMPDxYCHwBoZGQCB" +
                                "A8PFgIfAGhkZAIFDw8WAh8AaGRkGAEFHl9fQ29udHJvb" +
                                "HNSZXF1aXJlUG9zdEJhY2tLZXlfXxYBBQVpbWdHbykGG" +
                                "X8w9mWXTG2u7N%2BFEILNM9Vx"
                    )
                )
                assertThat(body, containsString("__VIEWSTATEGENERATOR=2E1E9F71"))
                assertThat(
                    body, containsString(
                        "__EVENTVALIDATION=%2FwEWCQLivviLCwLS8oWtCwLT8oWtCwLQ" +
                                "8oWtCwLS8rmtCwLT8rmtCwLQ8rmtCwLa9PCNDgKryeHr" +
                                "BlJ%2ByCHGEtqYGvk%2FByYes8959QuR"
                    )
                )
                assertThat(body, containsString("dlBillType=2R"))

                server.takeRequest(1, TimeUnit.SECONDS)!! // takes the first page again
            }

            Then("journal is requested") {
                val rr = server.takeRequest(1, TimeUnit.SECONDS)!!
                val url = rr.requestUrl!!
                assertEquals(rr.method, "GET")
                assertEquals("/lis/journal.aspx", url.encodedPath)
                assertEquals("17", url.queryParameter("congress"))
                assertEquals("2R", url.queryParameter("session"))
                assertEquals("58", url.queryParameter("q"))
            }

            Then("journal is not null") {
                assertNotNull(actual)
            }
        }
    }
})