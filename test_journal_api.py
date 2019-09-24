import datetime

import responses

from chamber import Chamber
from congress import Congress
from journal import Journal
from journal_api import HouseJournalApi
from session import Session


@responses.activate
def test_house_fetch_one():
    responses.add(
        method=responses.GET,
        url="http://www.congress.gov.ph"
            "/legisdocs?v=journals&congress=17&session=1",
        body=open("test/resources/house/journal/17th-session1.html", "rb"),
        match_querystring=True)

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


@responses.activate
def test_house_fetch_all():
    responses.add(
        method=responses.GET,
        url="http://www.congress.gov.ph"
            "/legisdocs?v=journals&congress=17&session=1",
        body=open("test/resources/house/journal/17th-session1.html", "rb"))

    congress = Congress(17)
    session = Session(1, "regular")

    api = HouseJournalApi()
    journals = api.fetch(congress, session)
    total = 0
    for journal in journals:
        total += 1

    assert total == 97


@responses.activate
def test_house_fetch_with_offset():
    responses.add(
        method=responses.GET,
        url="http://www.congress.gov.ph"
            "/legisdocs?v=journals&congress=17&session=1",
        body=open("test/resources/house/journal/17th-session1.html", "rb"))

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
