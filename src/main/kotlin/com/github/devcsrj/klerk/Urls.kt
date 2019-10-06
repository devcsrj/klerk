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