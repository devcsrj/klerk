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
import org.bytedeco.opencv.global.opencv_imgcodecs.imwrite

internal class Debug : DoFn<
        KV<Journal, PageSlice>,
        KV<Journal, PageSlice>>() {

    @ProcessElement
    fun processElement(context: ProcessContext) {

        val element = context.element()
        val journal = element.key!!
        val slice = element.value

        val mat = slice.mat.toMat()
        imwrite("j${journal.number}-p${slice.page}-${slice.index}.png", mat)
        context.output(KV.of(journal, slice))
    }
}