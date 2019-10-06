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

data class Session(
    val number: Int,
    val type: Type
) : Serializable {

    override fun toString(): String {
        return "$type Session $number"
    }

    companion object {

        fun regular(number: Int) = Session(number, Type.REGULAR)
        fun special(number: Int) = Session(number, Type.SPECIAL)
    }

    enum class Type {
        REGULAR,
        SPECIAL
    }
}