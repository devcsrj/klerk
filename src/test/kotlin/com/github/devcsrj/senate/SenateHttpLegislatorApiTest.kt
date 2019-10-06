package com.github.devcsrj.senate

import com.github.devcsrj.klerk.Legislator
import com.github.devcsrj.klerk.enqueue
import com.github.devcsrj.klerk.legislator.SenateHttpLegislatorApi
import okhttp3.mockwebserver.MockWebServer
import org.codehaus.jackson.map.ObjectMapper
import org.codehaus.jackson.type.TypeReference
import org.hamcrest.CoreMatchers.`is`
import org.hamcrest.MatcherAssert.assertThat
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.gherkin.Feature

object SenateHttpLegislatorApiTest : Spek({

    val baseDir = "/senate/legislator"

    Feature("Legislator") {
        lateinit var server: MockWebServer
        lateinit var api: SenateHttpLegislatorApi
        lateinit var expected: Set<Legislator>
        lateinit var actual: Set<Legislator>

        beforeGroup {
            server = MockWebServer()
            api = SenateHttpLegislatorApi(server.url("/").toUri())
            val json = this::class.java.getResource( "$baseDir/ph-senator-list-oct-6-2019.json")
            expected = ObjectMapper().readValue(json, object : TypeReference<Set<Legislator>>(){})
        }

        afterGroup {
            server.shutdown()
        }

        Scenario("Fetch all PH senators") {
            Given("Page") {
                server.enqueue(200, "$baseDir/ph-senator-list-oct-6-2019.html")
            }

            When("fetch() is called") {
                actual = api.fetch()
            }

            Then("all expected PH senators are extracted") {
                assertThat(actual, `is`(expected))
            }
        }
    }
})