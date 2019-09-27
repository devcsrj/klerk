import datetime
import re
from abc import ABCMeta, abstractmethod
from typing import Iterator, Optional
from urllib.parse import urlparse, parse_qs

import bs4
import requests
from bs4 import BeautifulSoup, Tag
from itertools import chain

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

        with requests.get(self._base_uri + self._path, params={
            "v": "journals",
            "congress": congress.number,
            "session": session.number
        }) as response:
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


class SenateJournalApi(JournalApi):
    """
    Fetches journals from
    http://www.senate.gov.ph/lis/leg_sys.aspx?congress=17&type=journal
    """

    def __init__(self,
                 base_uri="http://www.senate.gov.ph"):
        self._base_uri = base_uri
        self._path = "/lis/leg_sys.aspx"

    def fetch(self,
              congress: Congress,
              session: Session,
              offset=0) -> Iterator[Journal]:
        if offset > 0:
            raise NotImplementedError

        rs = self._change_congress(congress, session)
        return self._read_journals(rs, congress, session, 99)

    def _change_congress(self,
                         congress: Congress,
                         session: Session) -> requests.Session():
        rs = requests.Session()
        # ASP.NET pages are a nightmare to crawl, as they add additional
        # client state details in the pages.  So first we fetch the
        # "landing page"
        url = self._base_uri + self._path \
              + "?type=journal" \
              + "&congress=" + str(congress.number)
        with rs.get(url) as rp:
            document = BeautifulSoup(rp.text, features="html.parser")

        bill_type = str(session.number) + session.type[0].upper()

        # Then we extract the required state variables
        with rs.post(url, data={
            "__EVENTTARGET": "dlBillType",
            "__EVENTARGUMENT": "",
            "__VIEWSTATE": document.select_one("#__VIEWSTATE").get("value"),
            "__VIEWSTATEGENERATOR":
                document.select_one("#__VIEWSTATEGENERATOR").get("value"),
            "__EVENTVALIDATION":
                document.select_one("#__EVENTVALIDATION").get("value"),
            "dlBillType": bill_type
        }) as rp:
            pass

        return rs

    def _read_journals(self,
                       rs: requests.Session,
                       congress: Congress,
                       session: Session,
                       page=99) -> Iterator[Journal]:

        url = self._base_uri + self._path \
              + "?type=journal" \
              + "&congress=" + str(congress.number) \
              + "&p=" + str(page)
        with rs.get(url) as rp:
            document = BeautifulSoup(rp.text, features="html.parser")
        return self._read_journals_from_document(
            rs, congress, session, document)

    def _read_journals_from_document(self,
                                     rs: requests.Session,
                                     congress: Congress,
                                     session: Session,
                                     document: bs4) -> Iterator[Journal]:

        links = document.select("#lis_journal_table > div > * > a")
        previous_journals = self._read_journals_from_links(
            congress, session, links)

        previous = document.select_one("#pnl_NavTop > div > div > a")
        has_previous = previous.text == "Previous"

        if has_previous:
            href = previous.get("href")
            path = urlparse(self._base_uri + href)
            query_map = parse_qs(path.query)
            page = int(query_map["p"][0])
            next_journals = self._read_journals(
                rs, congress, session, page=page)
            return chain(previous_journals, next_journals)

        return previous_journals

    def _read_journals_from_links(self,
                                  congress: Congress,
                                  session: Session,
                                  links: Tag) -> Iterator[Journal]:
        for link in reversed(links):
            href = link.get("href")
            path = self._base_uri + "/lis/" + href
            yield self._read_journal(congress, session, path)

    def _read_journal(self,
                      congress: Congress,
                      session: Session,
                      url: str) -> Journal:

        with requests.get(url) as rp:
            document = BeautifulSoup(rp.text, features="html.parser")

        p = document.select_one("#content > div.lis_doctitle > p")
        number = p.text[len("Journal No. "):].strip()

        href = document.select_one("#lis_download > ul > li > a").get("href")
        uri = self._base_uri + href

        content = document.select_one("#content").text
        matcher = re.search(".*Date: (("
                            "January"
                            "|February"
                            "|March"
                            "|April"
                            "|May"
                            "|June"
                            "|July"
                            "|August"
                            "|September"
                            "|October"
                            "|November"
                            "|December) \\d{1,2}, \\d{4})", content)
        date_text = matcher.group(1)
        date = datetime.datetime.strptime(date_text, "%B %d, %Y").date()

        return Journal(
            chamber=Chamber.SENATE,
            congress=congress,
            session=session,
            number=int(number),
            date=date,
            document_uri=uri
        )
