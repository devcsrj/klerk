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

import java.io.Serializable

data class Session(
    val number: Int,
    val type: Type
) : Serializable {

    override fun toString(): String {
        return "${number.ordinal()} $type Session"
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