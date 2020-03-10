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

import com.github.devcsrj.klerk.*
import org.springframework.batch.core.Job
import org.springframework.batch.core.Step
import org.springframework.batch.core.configuration.annotation.JobBuilderFactory
import org.springframework.batch.core.configuration.annotation.StepBuilderFactory
import org.springframework.batch.core.launch.support.RunIdIncrementer
import org.springframework.batch.item.ItemReader
import org.springframework.batch.item.ItemWriter
import org.springframework.batch.item.file.LineMapper
import org.springframework.batch.item.file.builder.FlatFileItemReaderBuilder
import org.springframework.batch.item.file.builder.FlatFileItemWriterBuilder
import org.springframework.batch.item.file.transform.LineAggregator
import org.springframework.batch.item.support.IteratorItemReader
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.core.io.PathResource
import org.springframework.core.io.Resource


@Configuration
open class CollationConfig(
    private val jobBuilderFactory: JobBuilderFactory,
    private val stepBuilderFactory: StepBuilderFactory,
    private val props: KlerkProperties
) {

    @Bean
    internal open fun collateJournals(): Job {
        return jobBuilderFactory.get("collateJournalsJob")
            .incrementer(RunIdIncrementer())
            .start(collectRemoteJournalsStep())
            .next(downloadRemoteJournalsStep())
            .build()
    }

    @Bean
    internal open fun journalsResource(): Resource {
        return PathResource(props.outputDir.resolve("journals.jsonl"))
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
        val lineMapper = LineMapper { line, _ -> Journal.fromJson(line) }
        return FlatFileItemReaderBuilder<Journal>()
            .name("journalJsonlItemReader")
            .resource(journalsResource())
            .lineMapper(lineMapper)
            .build()
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
        val lineAggregator = LineAggregator<Journal> { item -> item.asJson() }
        return FlatFileItemWriterBuilder<Journal>()
            .name("journalJsonlItemWriter")
            .resource(journalsResource())
            .lineAggregator(lineAggregator)
            .build()
    }

    @Bean
    internal open fun journalApiItemReader(): ItemReader<Journal> {
        var iterator: Sequence<Journal> = emptySequence()

        val senateApi = SenateHttpJournalApi(props.senate.uri)
        val senateRequests = props.senate.congress
        for (request in senateRequests) {
            iterator += request.value
                .map { senateApi.fetch(request.key, it) }
                .reduce { left, right -> left + right }
        }

        val houseApi = HouseHttpJournalApi(props.house.uri)
        val houseRequests = props.house.congress
        for (request in houseRequests) {
            iterator += request.value
                .map { houseApi.fetch(request.key, it) }
                .reduce { left, right -> left + right }
        }
        return IteratorItemReader(iterator.iterator())
    }
}