import datetime
from abc import ABCMeta, abstractmethod
from typing import Iterator, Optional

import requests
from bs4 import BeautifulSoup, Tag

from chamber import Chamber
from congress import Congress
from journal import Journal
from session import Session


class JournalApi:
    __metaclass__ = ABCMeta

    @abstractmethod
    def fetch(self,
              congress: Congress,
              session: Session,
              offset=0) -> Iterator[Journal]:
        raise NotImplementedError


class HouseJournalApi(JournalApi):
    """
    Fetches journals from http://www.congress.gov.ph/legisdocs/?v=journals
    """

    def __init__(self,
                 base_uri="http://www.congress.gov.ph"):
        self._base_uri = base_uri
        self._path = "/legisdocs"

    def fetch(self,
              congress: Congress,
              session: Session,
              offset=0) -> Iterator[Journal]:
        if offset < 0:
            raise ValueError("Offset should at least be 0. Got: " + str(offset))

        response = requests.get(self._base_uri + self._path, params={
            "v": "journals",
            "congress": congress.number,
            "session": session.number
        })
        document = BeautifulSoup(response.text, features="html.parser")
        trs = document.select("table > tbody > tr")
        for tr in reversed(trs):
            journal = self._read_journal(congress, session, tr)
            if journal is None or journal.number <= offset:
                continue
            yield journal

    def _read_journal(self,
                      congress: Congress,
                      session: Session,
                      tr: Tag) -> Optional[Journal]:
        if tr.name != "tr":
            raise ValueError("Expecting a <tr> but got " + tr.name)

        children = tr.children
        first = next(children)
        if type(first) is not Tag:
            return None

        prefix = "Journal No. "
        label: str = first.text
        if not label.startswith(prefix):
            return None  # We skip special sessions for now
        number = first.text[len(prefix):].strip()

        second: Tag = next(children)
        date = datetime.datetime.strptime(second.text, "%Y-%m-%d").date()

        third: Tag = next(children)
        href = third.select_one("a").get("href")
        uri = self._base_uri + "/" + href[3:].strip()

        return Journal(
            chamber=Chamber.HOUSE,
            congress=congress,
            session=session,
            number=int(number),
            date=date,
            document_uri=uri
        )
