import argparse
import logging
from dataclasses import asdict

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from itertools import chain
from jsonlines import jsonlines

from congress import Congress
from journal_api import HouseJournalApi, SenateJournalApi
from session import Session


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


class DoWriteRecordsFn(beam.DoFn):
    """
    Calls asdict() on each element and writes it
    as json lines
    """

    def __init__(self, output):
        beam.DoFn.__init__(self)
        self._writer = jsonlines.open(output, mode="w")

    def process(self, element, *args, **kwargs):
        self._writer.write(element.asdict())


class DoLogRecordsFn(beam.DoFn):
    """
    Simply prints the element to the console
    """

    def process(self, element, *args, **kwargs):
        print(element)
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
    known_args, pipeline_args = parser.parse_known_args(argv)

    congresses = known_args.congress.split(",")

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    with beam.Pipeline(options=pipeline_options) as p:
        (p
         | 'prepare' >> beam.Create(congresses)
         | 'congress' >> beam.Map(lambda c: Congress(int(c)))
         | 'fetch' >> beam.ParDo(DoFetchJournalFn())
         | 'print' >> beam.ParDo(DoLogRecordsFn())
         | 'write' >> beam.ParDo(DoWriteRecordsFn("dist/journals.jsonl"))
         )

        result = p.run()
        result.wait_until_finish()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
