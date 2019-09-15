package com.github.devcsrj.klerk


data class Congress(val number: Int) {

    init {
        require(number > 0) { "Number must be > 0" }
    }

    override fun toString(): String {
        val lastDigit = number % 10
        val lastTwoDigits = number % 100

        //Returns "th" on "teen" values with the last 2 digits being between 10 and 20
        if (lastTwoDigits in 10..20) {
            return number.toString() + "th Congress"
        }

        //Returns appropriate suffix on non-"teen" values
        return when (lastDigit) {
            1 -> number.toString() + "st Congress"
            2 -> number.toString() + "nd Congress"
            3 -> number.toString() + "rd Congress"
            else -> number.toString() + "th Congress"
        }
    }
}