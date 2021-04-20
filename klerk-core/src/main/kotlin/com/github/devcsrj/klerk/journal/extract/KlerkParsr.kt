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
            HeaderFooterDetection(),
            ReadingOrderDetection(
                minColumnWidthInPagePercent = 30.0
            ),
            WordsToLineNew,
            LinesToParagraph(),
            PageNumberDetection,
            HierarchyDetection
        ),
        extractor = Extractor(
            pdfExtractor = PdfMiner,
            ocrExtractor = Tesseract,
            languages = setOf("eng")
        ),
        output = Output(
            granularity = Output.Granularity.WORD,
            includeMarginals = false,
            formats = setOf(Json, Text)
        )
    )
}