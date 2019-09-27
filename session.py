from dataclasses import dataclass


@dataclass
class Session:
    """
    Convened by the Congress, as defined in the Constitution.

    Regular sessions are convened every year beginning on the
    4th Monday of July. A regular session can last until thirty
    days before the opening of its next regular session in the
    succeeding year.

    The President may, however, call special sessions which are
    usually held between regular legislative sessions to handle
    emergencies or urgent matters.
    """

    number: int
    type: str

    def __str__(self):
        return "{} session {}".format(self.type, self.number)
