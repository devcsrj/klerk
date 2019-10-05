import datetime
from dataclasses import dataclass

from chamber import Chamber
from congress import Congress
from session import Session


@dataclass
class Journal:
    """
    The audit trail of a Congress session
    """

    chamber: Chamber
    congress: Congress
    session: Session
    number: int
    date: datetime.date
    document_uri: str
