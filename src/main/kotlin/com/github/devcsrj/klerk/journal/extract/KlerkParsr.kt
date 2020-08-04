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
package com.github.devcsrj.klerk.journal.extract

import com.github.devcsrj.docparsr.*

internal object KlerkParsr {

    val CONFIG = Configuration(
        version = "0.9",
        cleaners = setOf(
            OutOfPageRemoval,
            WhitespaceRemoval(),
            RedundancyDetection(),
            HeaderFooterDetection(),
            ReadingOrderDetection(
                minColumnWidthInPagePercent = 30.0
            ),
            WordsToLineNew,
            LinesToParagraph(),
            MlHeadingDetection,
            PageNumberDetection,
            HierarchyDetection,
            RegexMatcher(
                queries = setOf(
                    RegexMatcher.Query(FirstReading.HEADING_LABEL, FirstReading.HEADING_REGEX),
                    RegexMatcher.Query(FirstReading.BILL_LINE_LABEL, FirstReading.BILL_LINE_REGEX),
                    RegexMatcher.Query(FirstReading.INTRODUCER_LABEL, FirstReading.INTRODUCER_REGEX),
                    RegexMatcher.Query(FirstReading.RECEIVING_COMMITTEE_LABEL, FirstReading.RECEIVING_COMMITTEE_REGEX)
                )
            )
        ),
        extractor = Extractor(
            pdfExtractor = PdfMiner,
            ocrExtractor = Tesseract,
            languages = setOf("eng")
        ),
        output = Output(
            granularity = Output.Granularity.WORD,
            includeMarginals = false,
            formats = setOf(Json)
        )
    )

    object FirstReading {

        private const val PREFIX = "FIRST_READING::"

        const val HEADING_LABEL = "${PREFIX}HEADING"
        const val HEADING_REGEX = "BILLS ON FIRST READING"

        const val BILL_LINE_LABEL = "${PREFIX}BILL"
        const val BILL_LINE_REGEX = "Bill No\\. (\\d+), entitled"

        const val INTRODUCER_LABEL = "${PREFIX}INTRODUCER"
        const val INTRODUCER_REGEX = "Introduced by (.+)"

        const val RECEIVING_COMMITTEE_LABEL = "${PREFIX}RECEIVING"
        const val RECEIVING_COMMITTEE_REGEX = "To the Committees on (.+)"
    }
}