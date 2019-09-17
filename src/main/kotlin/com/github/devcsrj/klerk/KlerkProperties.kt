package com.github.devcsrj.klerk

import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.context.annotation.Configuration
import java.net.URI
import java.nio.file.Path

@Configuration
@ConfigurationProperties("klerk")
class KlerkProperties {

    lateinit var outputDir: Path
    var house = House()
    var senate = Senate()

    class House {
        lateinit var baseUri: URI
    }

    class Senate {
        lateinit var baseUri: URI
    }
}