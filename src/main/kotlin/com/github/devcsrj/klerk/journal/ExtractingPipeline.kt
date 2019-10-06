package com.github.devcsrj.klerk.journal

import org.apache.beam.sdk.Pipeline
import org.apache.beam.sdk.options.Description
import org.apache.beam.sdk.options.PipelineOptions
import org.apache.beam.sdk.options.PipelineOptionsFactory
import org.apache.beam.sdk.options.Validation
import org.apache.beam.sdk.transforms.Create
import org.apache.beam.sdk.transforms.DoFn
import org.apache.beam.sdk.transforms.ParDo
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.text.PDFTextStripperByArea
import org.slf4j.LoggerFactory
import java.awt.Rectangle
import java.io.File


class ExtractingPipeline {

    companion object {

        private val logger = LoggerFactory.getLogger(ExtractingPipeline::class.java)
    }

    /**
     * Extracts the text from a PDF file and passes the written file.
     */
    class Extract : DoFn<File, File>() {

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
                    // This dimension skips the text in the borders
                    val rectangle = Rectangle(21, 77, 550, 727)

                    val stripper = PDFTextStripperByArea()
                    stripper.addRegion("content", rectangle)
                    for (page in pdf.pages) {
                        stripper.extractRegions(page)
                        val text = stripper.getTextForRegion("content")
                        writer.write(text)
                    }
                }
            }
            outputReceiver.output(output)
        }
    }

    interface Options : PipelineOptions {

        @Description("The directory where to look for journal PDF")
        @Validation.Required
        fun getDir(): String

        fun setDir(dir: String)
    }
}

fun main(args: Array<String>) {
    val options = PipelineOptionsFactory
        .fromArgs(*args)
        .withValidation()
        .`as`(ExtractingPipeline.Options::class.java)

    val regex = Regex("journal-\\d+\\.pdf\$")
    val src = File(options.getDir())
        .walkTopDown()
        .filter { it.name.matches(regex) }
        .toList()

    val pipeline = Pipeline.create(options)

    pipeline
        .apply("List", Create.of(src))
        .apply("Extract", ParDo.of(ExtractingPipeline.Extract()))

    pipeline.run().waitUntilFinish()
}