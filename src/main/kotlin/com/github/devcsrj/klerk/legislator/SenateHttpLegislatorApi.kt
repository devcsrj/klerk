/**
 * Klerk
 * Copyright (C) 2019 Reijhanniel Jearl Campos
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package com.github.devcsrj.klerk.legislator

import com.github.devcsrj.klerk.Legislator
import okhttp3.OkHttpClient
import okhttp3.Request
import org.jsoup.internal.StringUtil
import org.jsoup.nodes.Element
import org.jsoup.parser.Parser
import java.net.URI

class SenateHttpLegislatorApi(private val url: URI) : LegislatorApi {

    private val client = OkHttpClient()
    private val parser = Parser.htmlParser()

    constructor() : this(URI.create("http://www.senate.gov.ph/senlist.asp"))

    override fun fetch(): Set<Legislator> {
        val request = Request.Builder().get().url(url.toURL()).build()
        val document = client.newCall(request).execute().use { response ->
            response.body.use { body ->
                val reader = body!!.byteStream().bufferedReader()
                parser.parseInput(reader, url.toString())
            }
        }

        val legislators = hashSetOf<Legislator>()

        val tableTags = document.select("table[width]")
        for (table in tableTags) {
            val pTags = table.select("tr > td.senatorlist> p")
            for (p in pTags) {
                p.selectFirst("a")?.let { a ->
                    addLegislator(legislators, a)
                }

                val hasH2MidClass = p.attr("class") == "h2_mid"
                val hasRightAlign = p.attr("align") == "right"
                val isTextRepPh = p.text() == "Republic of the Philippines"
                if (hasH2MidClass || hasRightAlign || isTextRepPh) {
                    continue
                }

                addLegislator(legislators, p)
            }

            val smallTags = table.select("table tr > td.senatorlist > small")
            for (small in smallTags) {
                addLegislator(legislators, small.parents()[0])
            }
        }
        return legislators
    }

    private fun addLegislator(names: HashSet<Legislator>, element: Element) {
        var name = element.ownText()
        name = StringUtil.normaliseWhitespace(name)
        name = name.replace("*", "")

        if (name.isNullOrEmpty()) {
            return
        }
        names.add(Legislator(name))
    }
}