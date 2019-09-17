package com.github.devcsrj.klerk

import org.springframework.boot.SpringApplication
import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.context.annotation.Bean

@SpringBootApplication
class Klerk {

    fun main(args: Array<String>) {
        SpringApplication.run(Klerk::class.java, *args)
    }

    @Bean
    fun listOfCongress(): List<Congress> {
        return listOf(
            Congress(18)
        )
    }
}