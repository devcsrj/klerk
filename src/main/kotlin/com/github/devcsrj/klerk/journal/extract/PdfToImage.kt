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

import org.apache.beam.sdk.transforms.DoFn
import org.apache.pdfbox.io.MemoryUsageSetting
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.rendering.ImageType
import org.apache.pdfbox.rendering.PDFRenderer
import org.apache.pdfbox.tools.imageio.ImageIOUtil
import org.slf4j.LoggerFactory
import java.io.File

/**
 * Renders each page of the PDF as PNG files.
 *
 * Files are named in the form of:
 * ```
 * - journal-1.pdf
 * - journal-1-p1.png
 * - journal-1-p2.png
 * - journal-1-p3.png
 * - ...
 * ```
 */
internal class PdfToImage : DoFn<File, File>() {

    private val logger = LoggerFactory.getLogger(PdfToImage::class.java)

    @ProcessElement
    fun processElement(
        @Element file: File,
        outputReceiver: OutputReceiver<File>
    ) {

        logger.info("📷 $file")
        val pages = renderPages(file)
        for (page in pages) {
            outputReceiver.output(page)
        }
    }

    private fun renderPages(file: File): Collection<File> {
        val prefix = file.nameWithoutExtension
        val dir = file.parentFile
        val memory = MemoryUsageSetting.setupTempFileOnly()
        val files = mutableListOf<File>()
        PDDocument.load(file, memory).use { pdf ->
            pdf.resourceCache = null // We're consuming too much memory

            val renderer = PDFRenderer(pdf)
            for (page in 0 until pdf.numberOfPages) {
                val png = dir.resolve("$prefix-p$page.png")
                if (!png.isFile) {
                    val img = renderer.renderImageWithDPI(page, 144F, ImageType.GRAY)
                    png.outputStream().use { sink ->
                        ImageIOUtil.writeImage(img, "png", sink)
                    }
                    img.flush()
                }
                files.add(png)
            }
        }
        return files
    }
}