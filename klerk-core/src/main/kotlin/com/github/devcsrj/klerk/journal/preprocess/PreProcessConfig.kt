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
    internal open fun pdfPagesToImagesStep(): Step {
        val reader = IteratorItemReader(journalRepository.iterator())
        val writer = PdfPagesToImagesWriter(journalRepository)

        return stepBuilderFactory["pdfPagesToImages"]
            .chunk<Journal, Journal>(5)
            .reader(reader)
            .writer(writer)
            .build()
    }

    @Bean
    internal open fun detectLayoutStep(): Step {
        val reader = IteratorItemReader(journalRepository.iterator())
        val writer = LayoutDetectingWriter(journalRepository)

        return stepBuilderFactory["detectLayout"]
            .chunk<Journal, Journal>(5)
            .reader(reader)
            .writer(writer)
            .build()
    }
}