package com.github.devcsrj.klerk

data class Session(
    val number: Int,
    val type: Type
) {

    companion object {

        fun regular(number: Int) = Session(number, Type.REGULAR)
        fun special(number: Int) = Session(number, Type.SPECIAL)
    }

    enum class Type {
        REGULAR,
        SPECIAL
    }
}