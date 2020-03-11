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
package com.github.devcsrj.klerk.journal.collate

import com.github.devcsrj.klerk.journal.Journal
import com.github.devcsrj.klerk.journal.JournalAssets
import com.github.devcsrj.klerk.journal.JournalRepository
import me.tongfei.progressbar.DelegatingProgressBarConsumer
import me.tongfei.progressbar.ProgressBarBuilder
import okhttp3.*
import okio.*
import org.slf4j.LoggerFactory
import org.springframework.batch.item.ItemWriter
import java.io.IOException
import java.nio.file.Files
import java.time.Duration

/**
 * Downloads [Journal#documentUri] to the provided [repository].
 */
internal class JournalPdfItemWriter(
    private val repository: JournalRepository
) : ItemWriter<Journal> {

    private val logger = LoggerFactory.getLogger(JournalPdfItemWriter::class.java)
    private val httpClient = OkHttpClient.Builder()
        .readTimeout(Duration.ofSeconds(30L))
        .addNetworkInterceptor(ProgressInterceptor())
        .build()

    override fun write(items: MutableList<out Journal>) {
        items.forEach(this::write)
    }

    private fun write(item: Journal) {
        logger.info("Downloading '$item'...")
        val assets = repository.assets(item)
        val pdf = assets.file(JournalAssets.DOCUMENT)
        if (Files.exists(pdf))
            return

        val request = Request.Builder()
            .get().url(item.documentUri.toURL())
            .build()
        httpClient.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                logger.warn("Failed to download '{}' ({})", item, response.code)
            }
            Files.newOutputStream(pdf).use { sink ->
                response.body?.byteStream()?.copyTo(sink)
            }
        }
    }

    /**
     * Take directly from https://github.com/square/okhttp/blob/master/samples/guide/src/main/java/okhttp3/recipes/Progress.java
     */
    private class ProgressInterceptor : Interceptor {
        override fun intercept(chain: Interceptor.Chain): Response {
            val request = chain.request()
            val originalResponse: Response = chain.proceed(request);
            val filename = originalResponse
                .header("Content-Disposition")
                ?.substringAfter("filename=")
                ?.substringBefore(";") ?: request.url.pathSegments.last()
            val body = ProgressResponseBody(
                originalResponse.body!!,
                ProgressBarListener(filename)
            )
            return originalResponse.newBuilder()
                .body(body)
                .build();
        }
    }

    private class ProgressResponseBody internal constructor(
        private val responseBody: ResponseBody,
        private val progressListener: ProgressListener
    ) : ResponseBody() {

        private var bufferedSource: BufferedSource? = null

        override fun contentType(): MediaType? = responseBody.contentType()
        override fun contentLength(): Long = responseBody.contentLength()
        override fun source(): BufferedSource {
            if (bufferedSource == null) {
                bufferedSource = source(responseBody.source()).buffer()
            }
            return bufferedSource!!
        }

        private fun source(source: Source): Source {
            return object : ForwardingSource(source) {
                var totalBytesRead = 0L

                @Throws(IOException::class)
                override fun read(sink: Buffer, byteCount: Long): Long {
                    val bytesRead: Long = super.read(sink, byteCount)
                    // read() returns the number of bytes read, or -1 if this source is exhausted.
                    totalBytesRead += if (bytesRead != -1L) bytesRead else 0
                    progressListener.update(totalBytesRead, responseBody.contentLength(), bytesRead == -1L)
                    return bytesRead
                }
            }
        }
    }

    private interface ProgressListener {
        fun update(bytesRead: Long, contentLength: Long, done: Boolean)
    }

    private class ProgressBarListener(private val filename: String) : ProgressListener {

        private val logger = LoggerFactory.getLogger(JournalPdfItemWriter::class.java)
        private val bar = ProgressBarBuilder()
            .setTaskName("Download '$filename'")
            .setConsumer(DelegatingProgressBarConsumer(logger::info))
            .setInitialMax(-1L)
            .setUnit("MB", 1024 * 1024)
            .setUpdateIntervalMillis(5 * 1000)
            .build()

        override fun update(bytesRead: Long, contentLength: Long, done: Boolean) {
            if (done) {
                bar.close()
            } else {
                bar.maxHint(contentLength)
                bar.stepTo(bytesRead)
            }
        }
    }
}
