#!/usr/bin/python3
def main():
    # Make sure cwd is in path
    import sys
    import os
    from .menu import empty_fridge
    sys.path.insert(1, os.path.abspath('.'))

    # import the correct class and use classmethod load
    fr = empty_fridge(sys.argv[0])

    # boot up!
    fr.bootstrap()

if __name__ == '__main__':
    main()