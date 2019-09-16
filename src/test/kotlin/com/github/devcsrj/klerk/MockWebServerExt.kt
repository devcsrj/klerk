package com.github.devcsrj.klerk

import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer

fun MockWebServer.enqueue(code: Int, classpath: String) {
    this.enqueue(
        MockResponse()
            .setResponseCode(code)
            .setBody(
                javaClass.getResourceAsStream(classpath)
                    .bufferedReader()
                    .readText()
            )
    )
}