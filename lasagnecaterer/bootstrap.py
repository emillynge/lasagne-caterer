#!/usr/bin/python3
def get_fridge():
    # Make sure cwd is in path
    import sys
    import os
    from lasagnecaterer.menu import empty_fridge
    sys.path.insert(1, os.path.abspath('.'))

    # import the correct class and use classmethod load
    return empty_fridge(sys.argv[0])


def main():
    fr = get_fridge()
    # boot up!
    fr.bootstrap()
    return fr

if __name__ == '__main__':
    fr = main()