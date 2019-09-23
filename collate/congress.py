class Congress:
    """
    The national legislature of the Philippines.

    It is a bicameral body consisting of the Senate (upper chamber),
    and the House of Representatives (lower chamber)
    """

    def __init__(self, number: int):
        if number <= 0:
            raise ValueError("Number must be  > 0")
        self.number = number

    def __str__(self):
        last_digit = self.number % 10
        last_two_digits = self.number % 100

        if last_two_digits in range(10, 21):
            return str(self.number) + "th Congress"

        if last_digit == 1:
            return str(self.number) + "st Congress"
        elif last_digit == 2:
            return str(self.number) + "nd Congress"
        elif last_digit == 3:
            return str(self.number) + "rd Congress"
        else:
            return str(last_digit) + "th Congress"
