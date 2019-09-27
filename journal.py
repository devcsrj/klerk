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

    def asdict(self) -> dict:
        return {
            "congress": self.congress.number,
            "chamber": str(self.chamber),
            "session": {
                "number": self.session.number,
                "type": self.session.type
            },
            "number": self.number,
            "date": self.date.strftime("%Y-%m-%d"),
            "document_uri": self.document_uri
        }
