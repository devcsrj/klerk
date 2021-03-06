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
package com.github.devcsrj.klerk

import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.context.annotation.Configuration
import java.net.URI
import java.nio.file.Path

@Configuration
@ConfigurationProperties(prefix = "klerk")
open class KlerkProperties {

    var senate: Senate = Senate()
    var house: House = House()
    lateinit var outputDir: Path
    lateinit var parsrUri: URI

    open class Senate {

        lateinit var uri: URI
        var congress: Map<Congress, List<Session>> = mutableMapOf()
    }

    open class House {
        lateinit var uri: URI
        var congress: Map<Congress, List<Session>> = mutableMapOf()
    }
}