package com.github.devcsrj.klerk.batch

import org.springframework.batch.item.ItemReader

class CompositeItemReader<T>(
    private val readers: List<ItemReader<T>>
) : ItemReader<T> {

    private val readersIterator = readers.iterator()
    private var lastReader = ItemReader<T> { null }

    override fun read(): T? {
        val next = lastReader.read()
        if (next != null) {
            return next
        }

        if (!readersIterator.hasNext()) {
            return null // end of reading
        }

        lastReader = readersIterator.next()
        return read()
    }
}