import datetime

import httpretty

from chamber import Chamber
from congress import Congress
from journal import Journal
from journal_api import HouseJournalApi
from session import Session


class TestJournalApi:

    @httpretty.activate
    def test_house_fetch_one(self):
        httpretty.register_uri(
            method=httpretty.GET,
            uri="http://www.congress.gov.ph/legisdocs/?v=journals",
            body=open("test/resources/17th-session1.html", "r").read())

        congress = Congress(17)
        session = Session(1, "regular")

        api = HouseJournalApi()
        journals = api.fetch(congress, session)

        actual = next(journals)
        assert actual == Journal(
            chamber=Chamber.HOUSE,
            congress=congress,
            session=session,
            number=1,
            date=datetime.date(2016, 7, 25),
            document_uri="http://www.congress.gov.ph/legisdocs/journals_17"
                         "/J1-1RS-20160725.pdf"
        )

        rr = httpretty.last_request()
        assert rr.path.startswith("/legisdocs")
        query_map = rr.querystring
        assert query_map["congress"][0] == "17"
        assert query_map["session"][0] == "1"
        assert query_map["v"][0] == "journals"

    @httpretty.activate
    def test_house_fetch_all(self):
        httpretty.register_uri(
            method=httpretty.GET,
            uri="http://www.congress.gov.ph/legisdocs/?v=journals",
            body=open("test/resources/17th-session1.html", "r").read())

        congress = Congress(17)
        session = Session(1, "regular")

        api = HouseJournalApi()
        journals = api.fetch(congress, session)
        total = 0
        for journal in journals:
            total += 1

        assert total == 97

    @httpretty.activate
    def test_house_fetch_with_offset(self):
        httpretty.register_uri(
            method=httpretty.GET,
            uri="http://www.congress.gov.ph/legisdocs/?v=journals",
            body=open("test/resources/17th-session1.html", "r").read())

        congress = Congress(17)
        session = Session(1, "regular")

        api = HouseJournalApi()
        journals = api.fetch(congress, session, offset=5)
        actual = next(journals)

        assert actual == Journal(
            chamber=Chamber.HOUSE,
            congress=congress,
            session=session,
            number=6,
            date=datetime.date(2016, 8, 3),
            document_uri="http://www.congress.gov.ph/legisdocs/journals_17"
                         "/J6-1RS-20160803.pdf"
        )
