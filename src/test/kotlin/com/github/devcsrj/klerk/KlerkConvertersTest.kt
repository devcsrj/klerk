/**
 * Copyright [2020] [Reijhanniel Jearl Campos]
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
import kotlin.test.assertEquals

object KlerkConvertersTest : Spek({

    group("CongressConverter") {
        val data = mapOf(
            "17" to Congress(17),
            "1" to Congress(1)
        )
        data.forEach { entry ->
            test("convert '${entry.key}'") {
                val actual = CongressConverter.convert(entry.key)
                assertEquals(entry.value, actual)
            }
        }
    }

    group("SessionConverter") {
        val data = mapOf(
            "1R" to Session.regular(1),
            "2S" to Session.special(2)
        )
        data.forEach { entry ->
            test("convert '${entry.key}'") {
                val actual = SessionConverter.convert(entry.key)
                assertEquals(entry.value, actual)
            }
        }
    }
})