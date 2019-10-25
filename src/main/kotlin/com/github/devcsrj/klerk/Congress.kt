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

import java.io.Serializable

/**
 * Congress the national legislature of the Philippines.
 *
 * It is a bicameral body consisting of the Senate (upper chamber),
 * and the House of Representatives (lower chamber)
 */
data class Congress(val number: Int) : Serializable {

    init {
        require(number > 0) { "Number must be > 0" }
    }

    override fun toString(): String {
        val lastDigit = number % 10
        val lastTwoDigits = number % 100

        //Returns "th" on "teen" values with the last 2 digits being between 10 and 20
        if (lastTwoDigits in 10..20) {
            return number.toString() + "th Congress"
        }

        //Returns appropriate suffix on non-"teen" values
        return when (lastDigit) {
            1 -> number.toString() + "st Congress"
            2 -> number.toString() + "nd Congress"
            3 -> number.toString() + "rd Congress"
            else -> number.toString() + "th Congress"
        }
    }
}