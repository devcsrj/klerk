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
package com.github.devcsrj.klerk.journal.extract

import com.github.devcsrj.klerk.Journal
import org.apache.beam.sdk.transforms.DoFn
import org.apache.beam.sdk.values.KV

/**
 * Skips sections according to some rules.
 *
 * Some of the captures sections are not readable, such as
 * logos, signatures, and borders. This function skips such
 * slices.
 */
internal class SkipSection : DoFn<
        KV<Journal, PageSlice>,
        KV<Journal, PageSlice>>() {

    @ProcessElement
    fun processElement(context: ProcessContext) {

        val element = context.element()
        val journal = element.key!!
        val slice = element.value

        if (slice.mat.cols < 3 || slice.mat.rows < 3)
            return // min width is 3

        context.output(KV.of(journal, slice))
    }
}