package com.github.devcsrj.klerk

data class Session(
    val number: Int,
    val type: Type
) {

    companion object {

        fun regular(number: Int) = Session(number, Type.REGULAR)
    }

    enum class Type {
        REGULAR,
        SPECIAL
    }
}