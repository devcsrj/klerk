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

    private fun renderPages(file: File): Sequence<File> {
        val prefix = file.nameWithoutExtension
        val dir = file.parentFile
        val memory = MemoryUsageSetting.setupMixed(512)
        return sequence {
            PDDocument.load(file, memory).use { pdf ->
                val renderer = PDFRenderer(pdf)
                for (page in 0 until pdf.numberOfPages) {
                    val png = dir.resolve("$prefix-p$page.png")
                    if (!png.isFile) {
                        val img = renderer.renderImageWithDPI(page, 300F, ImageType.GRAY)
                        png.outputStream().use { sink ->
                            ImageIOUtil.writeImage(img, "png", sink)
                        }
                    }
                    yield(png)
                }
            }
        }
    }
}