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
package com.github.devcsrj.klerk

import org.spekframework.spek2.Spek
import java.net.URI
import java.time.LocalDate
import java.time.Month
import kotlin.test.assertEquals

object JournalTest : Spek({

    test("asJson()") {
        val journal = Journal(
            chamber = Chamber.HOUSE,
            congress = Congress(17),
            session = Session.regular(1),
            number = 1,
            date = LocalDate.of(2016, Month.JULY, 25),
            documentUri = URI.create("http://example.com/journal.pdf")
        )
        val actual = journal.asJson()
        val expected =
            "{\"chamber\":\"HOUSE\",\"congress\":{\"number\":17},\"session\":{\"number\":1,\"type\":\"REGULAR\"},\"number\":1,\"date\":\"2016-07-25\",\"documentUri\":\"http://example.com/journal.pdf\"}"
        assertEquals(expected, actual)
    }

    test("fromJson()") {
        val source =
            "{\"chamber\":\"HOUSE\",\"congress\":{\"number\":17},\"session\":{\"number\":1,\"type\":\"REGULAR\"},\"number\":1,\"date\":\"2016-07-25\",\"documentUri\":\"http://example.com/journal.pdf\"}"
        val actual = Journal.fromJson(source)
        val expected = Journal(
            chamber = Chamber.HOUSE,
            congress = Congress(17),
            session = Session.regular(1),
            number = 1,
            date = LocalDate.of(2016, Month.JULY, 25),
            documentUri = URI.create("http://example.com/journal.pdf")
        )
        assertEquals(expected, actual)
    }
})