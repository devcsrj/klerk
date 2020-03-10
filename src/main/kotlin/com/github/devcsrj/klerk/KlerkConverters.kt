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
            'R' -> Session.regular(source[0].toInt())
            'S' -> Session.special(source[0].toInt())
            else -> throw UnsupportedOperationException("unexpected '${source[1]}', expecting 'R' or 'S'")
        }
    }
}
