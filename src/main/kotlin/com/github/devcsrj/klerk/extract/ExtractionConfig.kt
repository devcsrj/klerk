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
package com.github.devcsrj.klerk.extract

import com.github.devcsrj.docparsr.DocParsr
import com.github.devcsrj.docparsr.ParsingResult
import com.github.devcsrj.klerk.Journal
import com.github.devcsrj.klerk.KlerkProperties
import org.springframework.batch.core.Job
import org.springframework.batch.core.Step
import org.springframework.batch.core.configuration.annotation.JobBuilderFactory
import org.springframework.batch.core.configuration.annotation.StepBuilderFactory
import org.springframework.batch.core.launch.support.RunIdIncrementer
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import java.io.File

@Configuration
open class ExtractionConfig(
    private val jobBuilderFactory: JobBuilderFactory,
    private val stepBuilderFactory: StepBuilderFactory,
    private val props: KlerkProperties
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
        return stepBuilderFactory["parseJournals"]
            .chunk<File, Pair<Journal, ParsingResult>>(5)
            .reader(directoryPdfItemReader())
            .processor(journalProcessingProcessor())
            .writer(parsingResultItemWriter())
            .build()
    }

    @Bean
    internal open fun directoryPdfItemReader(): DirectoryPdfItemReader {
        return DirectoryPdfItemReader(props.outputDir)
    }

    @Bean
    internal open fun journalProcessingProcessor(): JournalParsingProcessor {
        return JournalParsingProcessor(DocParsr.create(props.parsrUri))
    }

    @Bean
    internal open fun parsingResultItemWriter(): ParsingResultItemWriter {
        return ParsingResultItemWriter(props.outputDir)
    }
}