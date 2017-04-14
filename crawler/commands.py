"""Main entrypoint for crawler commands."""

import argparse
import crawler.algorithms.commands as algorithms

def main():
    """Execute the crawler commandline interface."""

    parser = argparse.ArgumentParser(
        description="Crawl data sources for Python scripts.",
    )

    parser.add_argument(
        'source',
        type=str,
        help='Data source to download. Available options: algorithms',
    )

    parser.add_argument(
        '--out',
        type=str,
        help='File to store labeled syntax trees from the datasource'
    )

    args = parser.parse_args()

    if args.source.lower() == 'algorithms':
        fetch_func = algorithms.fetch
    else:
        raise Exception('Please provide a data source.')

    fetch_func(args.out)
