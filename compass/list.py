import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description='List the available test cases or machines')
    args = parser.parse_args(sys.argv[2:])
