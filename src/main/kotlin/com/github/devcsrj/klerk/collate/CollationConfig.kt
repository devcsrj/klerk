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

import com.fasterxml.jackson.databind.ObjectMapper
import com.github.devcsrj.klerk.*
import com.github.devcsrj.klerk.journal.HouseHttpJournalApi
import com.github.devcsrj.klerk.journal.SenateHttpJournalApi
import org.springframework.batch.core.Job
import org.springframework.batch.core.Step
import org.springframework.batch.core.configuration.annotation.JobBuilderFactory
import org.springframework.batch.core.configuration.annotation.StepBuilderFactory
import org.springframework.batch.core.launch.support.RunIdIncrementer
import org.springframework.batch.item.ItemReader
import org.springframework.batch.item.ItemWriter
import org.springframework.batch.item.json.JsonFileItemWriter
import org.springframework.batch.item.json.JsonObjectMarshaller
import org.springframework.batch.item.support.IteratorItemReader
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.core.io.PathResource
import java.util.*

@Configuration
open class CollationConfig(
    private val jobBuilderFactory: JobBuilderFactory,
    private val stepBuilderFactory: StepBuilderFactory
) {

    @Bean
    open fun collateJournals(collectRemoteJournalStep: Step): Job {
        return jobBuilderFactory.get("collateJournalsJob")
            .incrementer(RunIdIncrementer())
            .flow(collectRemoteJournalStep)
            .end()
            .build()
    }

    @Bean
    open fun collectRemoteJournalsStep(
        journalApiItemReader: ItemReader<Journal>,
        journalJsonItemWriter: ItemWriter<Journal>
    ): Step {
        return stepBuilderFactory["collectRemoteJournals"]
            .chunk<Journal, Journal>(10)
            .reader(journalApiItemReader)
            .writer(journalJsonItemWriter)
            .build()
    }

    @Bean
    open fun journalJsonItemWriter(
        props: KlerkProperties,
        objectMapper: ObjectMapper
    ): ItemWriter<Journal> {
        val json = PathResource(props.outputDir.resolve("journals.json"))
        return JsonFileItemWriter(
            json,
            JsonObjectMarshaller { `object` -> objectMapper.writeValueAsString(`object`) })
    }

    @Bean
    open fun journalApiItemReader(props: KlerkProperties): ItemReader<Journal> {
        var iterator: Iterator<Journal> = Collections.emptyIterator()

        val senateApi = SenateHttpJournalApi(props.senate.uri)
        val senateRequests = readCongressMap(props.senate.congress)
        for (request in senateRequests) {
            iterator += request.value
                .map { senateApi.fetch(request.key, it) }
                .reduce { left, right -> left + right }
        }

        val houseApi = HouseHttpJournalApi(props.house.uri)
        val houseRequests = readCongressMap(props.house.congress)
        for (request in houseRequests) {
            iterator += request.value
                .map { houseApi.fetch(request.key, it) }
                .reduce { left, right -> left + right }
        }
        return IteratorItemReader(iterator)
    }

    private fun readCongressMap(source: Map<Int, List<String>>): Map<Congress, List<Session>> {
        val map = mutableMapOf<Congress, List<Session>>()
        for (item in source) {
            val congress = Congress(item.key)
            val sessions = mutableListOf<Session>()
            for (session in item.value) {
                check(session.length == 2) {
                    "expecting 'Session' to be a 2-letter character like '1R' or '3S', but got $session"
                }
                val s = when (session[1]) {
                    'R' -> Session.regular(session[0].toInt())
                    'S' -> Session.special(session[0].toInt())
                    else -> throw UnsupportedOperationException("unexpected '$session', expecting 'R' or 'S'")
                }
                sessions.add(s)
            }
            map[congress] = sessions
        }
        return map
    }
}