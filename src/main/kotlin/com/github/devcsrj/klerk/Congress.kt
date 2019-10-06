/**
 * Klerk
 * Copyright (C) 2019 Reijhanniel Jearl Campos
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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