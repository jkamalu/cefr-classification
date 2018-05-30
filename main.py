import numpy as np
import math

from argparse import ArgumentParser








def main():
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--pattern", dest="pattern", help="your help text")
    args = parser.parse_args()
    print(args.pattern)
    main()