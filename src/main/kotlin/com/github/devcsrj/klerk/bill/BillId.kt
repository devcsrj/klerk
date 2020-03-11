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
package com.github.devcsrj.klerk.bill

import com.github.devcsrj.klerk.Chamber

data class BillId(val chamber: Chamber, val number: Int) {

    init {
        require(number > 0) {
            "Bill number must be > 0, but got '$number'"
        }
    }

    override fun toString(): String {
        return when (chamber) {
            Chamber.SENATE -> "SBN-$number"
            Chamber.HOUSE -> "HBN-$number"
        }
    }
}
