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
package com.github.devcsrj.klerk.collate

import com.github.devcsrj.klerk.Journal
import com.github.devcsrj.klerk.KlerkProperties
import org.springframework.batch.core.Job
import org.springframework.batch.core.Step
import org.springframework.batch.core.configuration.annotation.JobBuilderFactory
import org.springframework.batch.core.configuration.annotation.StepBuilderFactory
import org.springframework.batch.core.launch.support.RunIdIncrementer
import org.springframework.batch.item.ItemReader
import org.springframework.batch.item.ItemWriter
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration


@Configuration
open class CollationConfig(
    private val jobBuilderFactory: JobBuilderFactory,
    private val stepBuilderFactory: StepBuilderFactory,
    private val props: KlerkProperties
) {

    @Bean
    internal open fun collateJournals(): Job {
        return jobBuilderFactory.get("collateJournals")
            .incrementer(RunIdIncrementer())
            .start(collectRemoteJournalsStep())
            .next(downloadRemoteJournalsStep())
            .build()
    }

    @Bean
    internal open fun downloadRemoteJournalsStep(): Step {
        return stepBuilderFactory["downloadRemoteJournals"]
            .chunk<Journal, Journal>(10)
            .reader(journalResourceItemReader())
            .writer(journalPdfItemWriter())
            .build()
    }

    @Bean
    internal open fun journalResourceItemReader(): ItemReader<Journal> {
        return JournalInfoItemReader(props.outputDir)
    }

    @Bean
    internal open fun journalPdfItemWriter(): ItemWriter<Journal> {
        return JournalPdfItemWriter(props.outputDir)
    }

    @Bean
    internal open fun collectRemoteJournalsStep(): Step {
        return stepBuilderFactory["collectRemoteJournals"]
            .chunk<Journal, Journal>(10)
            .reader(journalApiItemReader())
            .writer(journalResourceItemWriter())
            .build()
    }

    @Bean
    internal open fun journalResourceItemWriter(): ItemWriter<Journal> {
        return JournalInfoItemWriter(props.outputDir)
    }

    @Bean
    internal open fun journalApiItemReader(): ItemReader<Journal> {
        return LazyIteratorItemReader(lazy {
            var sequence: Sequence<Journal> = emptySequence()

            val senateApi = SenateHttpJournalApi(props.senate.uri)
            val senateRequests = props.senate.congress
            for (request in senateRequests) {
                sequence += request.value
                    .map { senateApi.fetch(request.key, it) }
                    .reduce { left, right -> left + right }
            }

            val houseApi = HouseHttpJournalApi(props.house.uri)
            val houseRequests = props.house.congress
            for (request in houseRequests) {
                sequence += request.value
                    .map { houseApi.fetch(request.key, it) }
                    .reduce { left, right -> left + right }
            }

            sequence.iterator()
        })
    }
}