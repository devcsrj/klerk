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
package com.github.devcsrj.klerk

import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.channels.Channels
import java.nio.file.Files
import java.nio.file.Path

/**
 * Downloads this url to the provided path
 */
internal fun URL.downloadTo(path: Path) {
    Files.createDirectories(path.parent)

    var conn: HttpURLConnection? = null
    try {
        conn = this.openConnection() as HttpURLConnection
        conn.requestMethod = "GET"
        conn.inputStream.use {
            val source = Channels.newChannel(it)
            FileOutputStream(path.toFile()).use { fd ->
                fd.channel.transferFrom(source, 0, Long.MAX_VALUE)
            }
        }
    } finally {
        conn?.disconnect()
    }
}