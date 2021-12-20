import json
import os
import tempfile

import bonobo
import cv2
import layoutparser.models as lpmodel
import layoutparser.ocr as lpocr
import numpy
from PIL import Image
from bonobo.config import use_context_processor
from pdf2image import convert_from_path

model = lpmodel.Detectron2LayoutModel(
    config_path='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',  # In model catalog
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},  # In model`label_map`
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5]  # Optional
)


def load_pdf_pages():
    input_file = os.environ['INPUT_FILE']
    if input_file is None:
        raise ValueError("Expecting an INPUT_FILE in the environment")

    with tempfile.TemporaryDirectory() as path:
        pages = convert_from_path(
            input_file,
            output_folder=path,
            dpi=300,
            fmt="jpeg")

        for i, page in enumerate(pages, start=1):
            yield {
                'page': i,
                'image': page,
            }


def as_opencv(*args):
    image = numpy.array(args[0]['image'])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    yield {
        **args[0],
        'image': image,
    }


def detect_layout(*args):
    image = Image.fromarray(args[0]['image'])
    yield {
        **args[0],
        'layout': model.detect(image)
    }


def detect_metadata(*args):
    _, width, _ = args[0]['image'].shape

    blocks = [x.block for x in args[0]['layout']]
    left_most = min(blocks, key=lambda b: b.x_1)
    right_most = max(blocks, key=lambda b: b.x_2)
    x_margin = left_most.x_1 + (width - right_most.x_2)

    yield {
        **args[0],
        'metadata': {
            'offset': left_most.x_1,
            'width': width - x_margin,
        }
    }


def ocr(*args):
    ocr_agent = lpocr.TesseractAgent()

    for region in args[0]['layout']:
        segment = region.crop_image(args[0]['image'])
        text = ocr_agent.detect(segment)
        yield {
            **args[0],
            'region': region,
            'text': text,
        }


def detect_column(*args):
    metadata = args[0]['metadata']

    x = args[0]['region'].block.center[0]
    x = x - metadata['offset']
    w = metadata['width']

    #         left          middle        right
    cols = [w / 3, w / 2, w / 3 * 2]
    computed = list(map(lambda c: abs(x - c), cols))
    col = computed.index(min(computed))

    yield {
        **args[0],
        'column': col,
    }


def normalize(*args):
    del args[0]['image']
    del args[0]['layout']
    del args[0]['metadata']
    yield {
     **args[0],
     'region': args[0]['region'].to_dict(),
    }


def with_opened_file(self, context):
    input_file = os.environ['INPUT_FILE']
    output_file = input_file[:input_file.index('.')] + '.jsonl'  # same file name, but jsonl
    output_file = output_file[output_file.index(os.sep) + 1:]  # strip leading prefix of path
    output_file = 'output' + os.sep + output_file
    with open(output_file, 'w+') as f:
        yield f


@use_context_processor(with_opened_file)
def write_jsonl(f, *args):
    json.dump(args[0], f)
    f.write('\n')


def get_graph(**options):
    """
    This function builds the graph that needs to be executed.

    :return: bonobo.Graph

    """

    graph = bonobo.Graph(
        load_pdf_pages,
        as_opencv,
        detect_layout,
        detect_metadata,
        ocr,
        detect_column,
        normalize,
        write_jsonl,
    )

    return graph


def get_services(**options):
    """
    This function builds the services dictionary, which is a simple dict of names-to-implementation used by bonobo
    for runtime injection.

    It will be used on top of the defaults provided by bonobo (fs, http, ...). You can override those defaults, or just
    let the framework define them. You can also define your own services and naming is up to you.

    :return: dict
    """
    return {}


# The __main__ block actually execute the graph.
if __name__ == '__main__':
    parser = bonobo.get_argument_parser()
    with bonobo.parse_args(parser) as options:
        bonobo.run(
            get_graph(**options),
            services=get_services(**options)
        )
