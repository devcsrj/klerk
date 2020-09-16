package com.github.devcsrj.klerk.journal.preprocess

import com.github.devcsrj.klerk.journal.Journal
import com.github.devcsrj.klerk.journal.JournalRepository
import org.springframework.batch.core.Job
import org.springframework.batch.core.Step
import org.springframework.batch.core.configuration.annotation.JobBuilderFactory
import org.springframework.batch.core.configuration.annotation.StepBuilderFactory
import org.springframework.batch.core.launch.support.RunIdIncrementer
import org.springframework.batch.item.support.IteratorItemReader
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
open class PreProcessConfig(
    private val jobBuilderFactory: JobBuilderFactory,
    private val stepBuilderFactory: StepBuilderFactory,
    private val journalRepository: JournalRepository
) {

    @Bean
    internal open fun preProcessJournals(): Job {
        return jobBuilderFactory.get("preProcessJournals")
            .incrementer(RunIdIncrementer())
            .start(pdfPagesToImagesStep())
            .build()
    }

    @Bean
    internal open fun pdfPagesToImagesStep() : Step{
        val reader = IteratorItemReader(journalRepository.iterator())
        val writer = PdfPagesToImagesWriter(journalRepository)

        return stepBuilderFactory["pdfPagesToImages"]
            .chunk<Journal, Journal>(5)
            .reader(reader)
            .writer(writer)
            .build()
    }
}