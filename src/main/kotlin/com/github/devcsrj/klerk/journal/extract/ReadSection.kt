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
import org.bytedeco.leptonica.global.lept.pixDestroy
import org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY
import org.bytedeco.opencv.global.opencv_imgproc.cvtColor
import org.bytedeco.opencv.opencv_core.Mat
import org.bytedeco.tesseract.ETEXT_DESC
import org.bytedeco.tesseract.TessBaseAPI
import org.bytedeco.tesseract.global.tesseract.RIL_PARA
import org.slf4j.LoggerFactory

/**
 * Reads each image slice as text
 */
internal class ReadSection : DoFn<
        KV<Journal, Iterable<@JvmWildcard PageSlice>>,
        KV<Journal, PageBlock>>() {

    private val logger = LoggerFactory.getLogger(ReadSection::class.java)

    @ProcessElement
    fun processElement(context: ProcessContext) {

        val element = context.element()
        val journal = element.key!!
        val blocks = readBlocks(element.value.sorted())
        for (block in blocks) {
            logger.info("🔎 $block ($journal)")
            context.output(KV.of(journal, block))
        }
    }

    private fun readBlocks(slices: Iterable<PageSlice>): List<PageBlock> {

        val tesseract = TessBaseAPI().apply {
            if (Init(".", "eng") != 0) {
                throw AssertionError("⚠️ Could not find OCR training data 'eng.traineddata'")
            }
        }
        val blocks = mutableListOf<PageBlock>()
        tesseract.use { api ->
            for (slice in slices) {
                logger.info("🔎 $slice")
                val block = readSlice(slice, api)
                blocks.add(block)
            }
        }
        return blocks
    }

    private fun readSlice(slice: PageSlice, api: TessBaseAPI): PageBlock {
        val mat = slice.mat.toMat()
        val grey = Mat()
        cvtColor(mat, grey, COLOR_BGR2GRAY)
        val pix = grey.use { it.toPix() }
        val string = StringBuilder()

        api.SetImage(pix)
        api.SetVariable("preserve_interword_spaces", "1")
        api.Recognize(ETEXT_DESC())
        val iterator = api.GetIterator()

        val level = RIL_PARA
        iterator.Begin()
        do {
            val str = iterator.GetUTF8Text(level)?.getString("UTF-8")
            if (!(str == null || str.isBlank())) {
                string.append(str)
                string.append("\n")
            }
        } while (iterator.Next(level))
        pixDestroy(pix)

        return PageBlock(slice.page, slice.index, slice.file, string.toString())
    }

}