import argparse

def argument_parser(epilog=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="path to dataset")
    parser.add_argument("--output_dir", type=str, help="path to segmentation reslut")
    return parser