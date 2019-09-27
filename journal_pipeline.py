import argparse
import json
import logging
from pathlib import Path

import apache_beam as beam
import requests
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from itertools import chain
from tqdm import tqdm

from congress import Congress
from journal import Journal
from journal_api import SenateJournalApi, HouseJournalApi
from session import Session


def _journal_dir(journal: Journal, base_dir: Path) -> Path:
    congress = str(journal.congress.number)
    session = journal.session.type + "-" + str(journal.session.number)
    return base_dir / congress / session


class DoPrintJournalFn(beam.DoFn):
    """
    Prints the current journal to the console
    """

    def process(self, element, *args, **kwargs):
        journal: Journal = element
        print("{} ({}) - {} Journal {}"
              .format(journal.congress, journal.session,
                      journal.chamber, journal.number))
        return [element]


class DoFetchJournalFn(beam.DoFn):
    """
    Downloads legislative session journals
    """

    def __init__(self):
        beam.DoFn.__init__(self)
        self.senate_api = SenateJournalApi()
        self.house_api = HouseJournalApi()

    def process(self, element, *args, **kwargs):
        """

        :param element: the congress to fetch journals from
        :return: an iterator of journals
        """
        congress: Congress = element
        sessions = [
            Session(1, "regular"),
            Session(2, "regular"),
            Session(3, "regular"),
        ]

        items = iter(())
        for session in sessions:
            jh = self.house_api.fetch(congress, session)
            js = self.senate_api.fetch(congress, session)
            items = chain(items, js, jh)

        return items


class DoWriteJournalFn(beam.DoFn):
    """
    Writes to json file under '$congress/journal-$number.json'
    """

    def __init__(self, output_dir: Path):
        beam.DoFn.__init__(self)
        if not output_dir.is_dir():
            raise ValueError("Expecting a dir, but got: " + str(output_dir))
        self._output_dir = output_dir

    def process(self, element, *args, **kwargs):
        journal: Journal = element
        name = "journal-{}.json".format(journal.number)
        file = _journal_dir(journal, self._output_dir) / name
        file.parent.mkdir(parents=True, exist_ok=True)

        with file.open("w") as f:
            json.dump(element.asdict(), f, indent=4)
        return [element]


class DoDownloadJournalFn(beam.DoFn):
    """
    Downloads pdf under '$congress/journal-$number.pdf'
    """

    def __init__(self, output_dir: Path):
        beam.DoFn.__init__(self)
        if not output_dir.is_dir():
            raise ValueError("Expecting a dir, but got: " + str(output_dir))
        self._output_dir = output_dir

    def process(self, element, *args, **kwargs):
        journal: Journal = element
        name = "journal-{}.pdf".format(journal.number)
        file = _journal_dir(journal, self._output_dir) / name
        file.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(journal.document_uri, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        with tqdm(total=total_size, unit='iB', unit_scale=True) as progress:
            with file.open("wb") as f:
                for data in response.iter_content(block_size):
                    progress.update(len(data))
                    f.write(data)

        return [element]


def run(argv=None):
    """
    Main entry point for extracting bill audit trail
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--congress',
                        default='17',
                        help='Comma separated value of the Nth Congress to '
                             'process')
    parser.add_argument('--output',
                        default="dist",
                        help='Output directory')

    known_args, pipeline_args = parser.parse_known_args(argv)

    congresses = known_args.congress.split(",")
    output_dir = Path(known_args.output)

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    with beam.Pipeline(options=pipeline_options) as p:
        (p
         | 'prepare' >> beam.Create(congresses)
         | 'congress' >> beam.Map(lambda c: Congress(int(c)))
         | 'fetch' >> beam.ParDo(DoFetchJournalFn())
         | 'print' >> beam.ParDo(DoPrintJournalFn())
         | 'write' >> beam.ParDo(DoWriteJournalFn(output_dir))
         | 'download' >> beam.ParDo(DoDownloadJournalFn(output_dir))
         )

        result = p.run()
        result.wait_until_finish()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
