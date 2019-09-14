package com.github.devcsrj.klerk

import org.springframework.boot.SpringApplication
import org.springframework.boot.autoconfigure.SpringBootApplication

@SpringBootApplication
class Klerk {

    fun main(args: Array<String>) {
        SpringApplication.run(Klerk::class.java, *args)
    }
}