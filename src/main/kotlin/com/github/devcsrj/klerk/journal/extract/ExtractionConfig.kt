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
package com.github.devcsrj.klerk.journal.extract

import com.github.devcsrj.docparsr.DocParsr
import com.github.devcsrj.docparsr.ParsingResult
import com.github.devcsrj.klerk.KlerkProperties
import com.github.devcsrj.klerk.journal.Assets
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
open class ExtractionConfig(
    private val jobBuilderFactory: JobBuilderFactory,
    private val stepBuilderFactory: StepBuilderFactory,
    private val props: KlerkProperties,
    private val journalRepository: JournalRepository
) {

    @Bean
    internal open fun extractJournals(): Job {
        return jobBuilderFactory.get("extractJournals")
            .incrementer(RunIdIncrementer())
            .start(parseJournalsStep())
            .build()
    }

    @Bean
    internal open fun parseJournalsStep(): Step {
        val reader = IteratorItemReader(journalRepository.iterator())
        val processor = JournalParsingProcessor(DocParsr.create(props.parsrUri), journalRepository)
        val writer = ParsingResultItemWriter()

        return stepBuilderFactory["parseJournals"]
            .chunk<Journal, Pair<Assets, ParsingResult>>(5)
            .reader(reader)
            .processor(processor)
            .writer(writer)
            .build()
    }
}
