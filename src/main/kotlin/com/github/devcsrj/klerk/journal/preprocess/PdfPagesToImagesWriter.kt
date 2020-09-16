package com.github.devcsrj.klerk.journal.preprocess

import com.github.devcsrj.klerk.journal.Journal
import com.github.devcsrj.klerk.journal.JournalAssets
import com.github.devcsrj.klerk.journal.JournalRepository
import com.github.devcsrj.klerk.journal.extract.FirstReadingProcessor
import org.apache.pdfbox.io.MemoryUsageSetting
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.rendering.ImageType
import org.apache.pdfbox.rendering.PDFRenderer
import org.apache.pdfbox.tools.imageio.ImageIOUtil
import org.slf4j.LoggerFactory
import org.springframework.batch.item.ItemProcessor
import org.springframework.batch.item.ItemWriter
import java.io.File
import java.nio.file.Files

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
internal class PdfPagesToImagesWriter(
    private val repository: JournalRepository
) : ItemWriter<Journal> {

    companion object {

        @JvmStatic
        private val logger = LoggerFactory.getLogger(PdfPagesToImagesWriter::class.java)
    }

    override fun write(items: MutableList<out Journal>) {
        items.forEach { process(it) }
    }

    private fun process(item: Journal) {
        val assets = repository.assets(item)
        val file = assets.file(JournalAssets.DOCUMENT)
        if (!Files.exists(file)) {
            return
        }

        val memory = MemoryUsageSetting.setupTempFileOnly()
        PDDocument.load(file.toFile(), memory).use { pdf ->
            pdf.resourceCache = null // We're consuming too much memory

            val renderer = PDFRenderer(pdf)
            for ((i, page) in (0 until pdf.numberOfPages).withIndex()) {
                logger.info("Rendering page $page of $item")

                val name = JournalAssets.DOCUMENT_PAGE(page, ".png")
                val png = assets.file(name).toFile()
                val img = renderer.renderImageWithDPI(page, 300F, ImageType.GRAY)
                png.outputStream().use { sink ->
                    ImageIOUtil.writeImage(img, "png", sink)
                }
                img.flush()
            }
        }
    }
}