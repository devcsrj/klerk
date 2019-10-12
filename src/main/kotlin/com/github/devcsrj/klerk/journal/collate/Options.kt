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
package com.github.devcsrj.klerk.journal.collate

import org.apache.beam.sdk.options.Description
import org.apache.beam.sdk.options.PipelineOptions
import org.apache.beam.sdk.options.Validation

internal interface Options : PipelineOptions {

    @Description("The congress # to fetch journals from (e.g., 17, 18)")
    @Validation.Required
    fun getInput(): List<Int>

    fun setInput(input: List<Int>)

    @Description("The destination directory to write the files to")
    fun getOutput(): String

    fun setOutput(output: String)
}