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
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.text.PDFTextStripper
import org.slf4j.LoggerFactory
import java.io.File

/**
 * Extracts the text from a PDF file and passes the written file.
 */
internal class PdfToTxt : DoFn<File, File>() {

    private val logger = LoggerFactory.getLogger(PdfToTxt::class.java)

    @ProcessElement
    fun processElement(
        @Element file: File,
        outputReceiver: OutputReceiver<File>
    ) {

        logger.info("👓 $file")
        val txt = file.nameWithoutExtension + ".txt"
        val output = file.parentFile.resolve(txt)
        PDDocument.load(file).use { pdf ->
            output.bufferedWriter().use { writer ->
                val stripper = PDFTextStripper()
                stripper.writeText(pdf, writer)
            }
        }
        outputReceiver.output(output)
    }
}