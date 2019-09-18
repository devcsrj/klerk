package com.github.devcsrj.klerk.journal

import com.github.devcsrj.klerk.*
import okhttp3.mockwebserver.MockWebServer
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.gherkin.Feature
import java.time.LocalDate
import java.time.Month
import java.util.concurrent.TimeUnit
import kotlin.test.assertEquals
import kotlin.test.assertTrue

object HouseHttpJournalApiTest : Spek({

    val baseDir = "/house/journal"
    val congress = Congress(17)
    val session = Session.regular(1)

    Feature("Journal") {
        lateinit var server: MockWebServer
        lateinit var api: HouseHttpJournalApi

        beforeGroup {
            server = MockWebServer()
            api = HouseHttpJournalApi(server.url("/").toUri())
        }

        afterGroup {
            server.shutdown()
        }

        Scenario("Fetch first journal") {

            lateinit var actual: Iterator<Journal>

            Given("Page") {
                server.enqueue(200, "$baseDir/17th-session1.html")
            }

            When("getAll() is called") {
                actual = api.fetch(congress, session)
            }

            Then("Journals are requested") {
                val rr = server.takeRequest(1, TimeUnit.SECONDS)!!
                val url = rr.requestUrl!!
                assertEquals(rr.method, "GET")
                assertEquals("/legisdocs", url.encodedPath)
                assertEquals("17", url.queryParameter("congress"))
                assertEquals("1", url.queryParameter("session"))
                assertEquals("journals", url.queryParameter("v"))
            }

            Then("Iterator has next") {
                assertTrue(actual.hasNext())
            }

            Then("it should get the journal") {
                val expected = Journal(
                    chamber = Chamber.HOUSE,
                    congress = congress,
                    session = Session.regular(1),
                    number = 1,
                    date = LocalDate.of(2016, Month.JULY, 25),
                    documentUri = server.url("/legisdocs/journals_17/J1-1RS-20160725.pdf").toUri()
                )
                assertEquals(expected, actual.next())
            }
        }

        Scenario("Fetch all journals") {

            var total: Int = 0

            Given("All pages") {
                server.enqueue(200, "$baseDir/17th-session1.html")
            }

            When("read all") {
                val it = api.fetch(congress, session)
                while (it.hasNext()) {
                    it.next()
                    total += 1
                }
            }

            Then("Total is correct") {
                assertEquals(97, total)
            }
        }

        Scenario("Fetch with offset") {

            lateinit var actual: Iterator<Journal>

            Given("Page") {
                server.enqueue(200, "$baseDir/17th-session1.html")
            }

            When("getAll() with offset 5 is called") {
                actual = api.fetch(congress, session, offset = 5)
            }

            Then("Iterator has next") {
                assertTrue(actual.hasNext())
            }

            Then("it should get the journal") {
                val expected = Journal(
                    chamber = Chamber.HOUSE,
                    congress = congress,
                    session = Session.regular(1),
                    number = 6,
                    date = LocalDate.of(2016, Month.AUGUST, 3),
                    documentUri = server.url("/legisdocs/journals_17/J6-1RS-20160803.pdf").toUri()
                )
                assertEquals(expected, actual.next())
            }
        }
    }

})