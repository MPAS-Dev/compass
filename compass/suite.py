import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Set up a regression test suite')
    args = parser.parse_args(sys.argv[2:])

