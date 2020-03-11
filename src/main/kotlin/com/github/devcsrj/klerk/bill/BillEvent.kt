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

import com.fasterxml.jackson.annotation.JsonFormat
import java.time.LocalDate

open class BillEvent(
    id: String,
    timestamp: LocalDate,
    type: String,
    billId: BillId
)

data class BillIntroducedEvent(
    val id: String,
    @JsonFormat(shape = JsonFormat.Shape.STRING, pattern = "yyyy-MM-dd")
    val timestamp: LocalDate,
    val billId: BillId,
    val title: String,
    val by: Author,
    val to: List<Committee>
) : BillEvent(id, timestamp, "bill-introduced", billId)