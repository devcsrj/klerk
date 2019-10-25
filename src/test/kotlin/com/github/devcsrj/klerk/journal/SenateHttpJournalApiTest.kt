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
package com.github.devcsrj.klerk.journal

import com.github.devcsrj.klerk.*
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.hamcrest.CoreMatchers.containsString
import org.hamcrest.MatcherAssert.assertThat
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.gherkin.Feature
import java.time.LocalDate
import java.time.Month
import java.util.concurrent.TimeUnit
import kotlin.test.assertEquals
import kotlin.test.assertTrue

object SenateHttpJournalApiTest : Spek({

    val baseDir = "/senate/journal"
    val congress = Congress(17)
    val session = Session.regular(1)

    Feature("Journal") {

        Scenario("Fetch first journal") {

            lateinit var server: MockWebServer
            lateinit var api: SenateHttpJournalApi
            lateinit var actual: Iterator<Journal>

            Given("Page") {
                server = MockWebServer()
                api = SenateHttpJournalApi(server.url("/").toUri())

                server.enqueue(200, "$baseDir/17th-regular_session1-p1.html") // takes the "landing page"
                server.enqueue(MockResponse().setResponseCode(302)) // requests a session change
                server.enqueue(200, "$baseDir/17th-regular_session1-p4.html")
                server.enqueue(200, "$baseDir/17th-regular_session1-p3.html")
                server.enqueue(200, "$baseDir/17th-regular_session1-p2.html")
                server.enqueue(200, "$baseDir/17th-regular_session1-p1.html")
                server.enqueue(200, "$baseDir/17th-regular_session1-journal1.html")
            }

            When("getAll() is called") {
                actual = api.fetch(congress, session)
            }

            Then("selected session is changed") {
                server.takeRequest(1, TimeUnit.SECONDS)!! // takes the landing page

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
                assertThat(body, containsString("dlBillType=1R"))
            }

            Then("journals are requested") {
                val rr = server.takeRequest(1, TimeUnit.SECONDS)!!
                val url = rr.requestUrl!!
                assertEquals(rr.method, "GET")
                assertEquals("/lis/leg_sys.aspx", url.encodedPath)
                assertEquals("17", url.queryParameter("congress"))
                assertEquals("99", url.queryParameter("p"))
                assertEquals("journal", url.queryParameter("type"))
            }

            Then("iterator has next()") {
                assertTrue(actual.hasNext())
            }
        }

        Scenario("Get one") {

            lateinit var server: MockWebServer
            lateinit var api: SenateHttpJournalApi
            lateinit var actual: Journal

            Given("Pages") {
                server = MockWebServer()
                api = SenateHttpJournalApi(server.url("/").toUri())

                server.enqueue(200, "$baseDir/17th-regular_session1-p1.html") // takes the "landing page"
                server.enqueue(MockResponse().setResponseCode(302)) // requests a session change
                server.enqueue(200, "$baseDir/17th-regular_session1-p4.html")
                server.enqueue(200, "$baseDir/17th-regular_session1-p3.html")
                server.enqueue(200, "$baseDir/17th-regular_session1-p2.html")
                server.enqueue(200, "$baseDir/17th-regular_session1-p1.html")
                server.enqueue(200, "$baseDir/17th-regular_session1-journal1.html")
            }

            When("getAll() is called") {
                actual = api.fetch(congress, session).next()
            }

            Then("journal is requested") {
                for (i in 6 downTo 1) {
                    server.takeRequest(1, TimeUnit.SECONDS)
                }

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
                    chamber = Chamber.SENATE,
                    congress = congress,
                    session = Session.regular(1),
                    number = 1,
                    date = LocalDate.of(2016, Month.JULY, 25),
                    documentUri = server.url("/lisdata/2376420165!.pdf").toUri()
                )
                assertEquals(expected, actual)
            }
        }
    }
})