package com.github.devcsrj.klerk.collate

import org.springframework.batch.item.ItemReader

internal class LazyIteratorItemReader<T>(val iterator: Lazy<Iterator<T>>) : ItemReader<T> {

    override fun read(): T? {
        val it = iterator.value
        return if (it.hasNext()) it.next() else null // end of data
    }
}