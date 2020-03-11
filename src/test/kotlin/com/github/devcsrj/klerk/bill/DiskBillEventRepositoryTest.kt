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
import com.google.common.jimfs.Jimfs
import org.hamcrest.MatcherAssert.assertThat
import org.hamcrest.Matchers.equalTo
import org.hamcrest.Matchers.hasSize
import org.spekframework.spek2.Spek
import java.nio.file.FileSystem
import java.nio.file.Files
import java.time.LocalDate

object DiskBillEventRepositoryTest : Spek({

    lateinit var fileSystem: FileSystem

    beforeEachTest {
        fileSystem = Jimfs.newFileSystem()
    }

    afterEachTest {
        fileSystem.close()
    }

    test("save") {
        val baseDir = fileSystem.getPath("klerk")
        Files.createDirectory(baseDir)

        val repo = DiskBillEventRepository(baseDir)
        repo.afterPropertiesSet()

        val event = BillTestEvent("id", LocalDate.of(2020, 1, 1), BillId(Chamber.SENATE, 999))
        repo.save(event)
        repo.save(event)

        val jsonl = baseDir.resolve("bill-events.jsonl")
        val lines = Files.readAllLines(jsonl)
        assertThat(lines, hasSize(2))
        assertThat(
            lines[0],
            equalTo("{\"id\":\"id\",\"timestamp\":[2020,1,1],\"billId\":{\"chamber\":\"SENATE\",\"number\":999}}")
        )
    }
})

data class BillTestEvent(
    val id: String,
    val timestamp: LocalDate,
    val billId: BillId
) : BillEvent(id, timestamp, "bill-test", billId)