from dataclasses import dataclass
from datetime import date, datetime
from bs4 import BeautifulSoup

import requests


@dataclass(frozen=True)
class Journal:
    congress: int
    session: int
    label: int
    date: date
    url: str

    def __str__(self):
        return f"{self.congress}-R{self.session}-{self.label} ({datetime.strftime(self.date, '%Y-%m-%d')})"

class CongressGov:
    """
    Class to handle the Congress.gov API

    See https://www.congress.gov.ph/legisdocs/?v=journals
    """

    def __init__(self):
        self.url = "https://www.congress.gov.ph/legisdocs/?v=journals"

    def get_journals(self, *, congress, session):
        """
        Get the list of journals for the given congress and session

        :param int congress: The congress number
        :param int session: The session number
        :return: List of Journal objects
        """
        page = requests.post(self.url, data={"v": "journals", "congress": congress, "session": session})
        page.raise_for_status()

        soup = BeautifulSoup(page.content, "html.parser")
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) != 3:
                continue

            is_header = cells[0].find("strong") is not None
            if is_header:
                continue

            label = cells[0].text
            try:
                # "2022-06-01", "2022-06-02"
                journal_date = datetime.strptime(cells[1].text, "%Y-%m-%d")
            except ValueError:
                continue

            url = cells[2].find("a")["href"]

            yield Journal(congress, session, label, journal_date, url)


if __name__ == "__main__":
    cg = CongressGov()
    for journal in cg.get_journals(congress=18, session=3):
        print(journal)