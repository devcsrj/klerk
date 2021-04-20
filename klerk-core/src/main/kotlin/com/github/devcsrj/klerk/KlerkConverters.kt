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

import org.springframework.boot.context.properties.ConfigurationPropertiesBinding
import org.springframework.core.convert.converter.Converter
import org.springframework.stereotype.Component

@Component
@ConfigurationPropertiesBinding
object CongressConverter : Converter<String, Congress> {
    override fun convert(source: String): Congress? {
        val num = source.toIntOrNull()
        require(num != null) {
            "expecting a numeric value, but got $source"
        }
        return Congress(num)
    }
}

@Component
@ConfigurationPropertiesBinding
object SessionConverter : Converter<String, Session> {
    override fun convert(source: String): Session? {
        require(source.length == 2) {
            "expecting a 2-letter character like '1R' or '3S', but got $source"
        }
        return when (source[1]) {
            'R' -> Session.regular(source[0].toString().toInt())
            'S' -> Session.special(source[0].toString().toInt())
            else -> throw UnsupportedOperationException("unexpected '${source[1]}', expecting 'R' or 'S'")
        }
    }
}
