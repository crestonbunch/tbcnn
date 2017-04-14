"""Utility functions for the crawler."""

import time
import json
import logging
import os.path

CONFIG_FILE = 'crawler/config.json'

def rate_limit(timeout=1.0):
    """Returns a response hook function which sleeps for `timeout` seconds if
    response is not cached."""
    def hook(response, *args, **kwargs):
        """Check if the result was cached, otherwise wait."""
        if not getattr(response, 'from_cache', False):
            logging.debug("Caching response for %s", response.url)
            time.sleep(timeout)
        else:
            logging.debug("Using cached response for %s", response.url)
        return response
    return hook

def github_auth(config_file):
    """Parse GitHub basic auth from the config file."""
    if not os.path.isfile(config_file):
        logging.warning("%s not found, not using GitHub authentication", config_file)
        return

    file_handler = open(config_file, 'r')
    contents = json.load(file_handler)
    return (contents['GITHUB_USERNAME'], contents['GITHUB_ACCESS_TOKEN'])

def flatten(t):
    """Flatten a 2-dimensional list. Returns a generator"""
    return [a for s in t for a in s]


request_auth_github = github_auth(CONFIG_FILE)
REQUEST_PERIOD = 3 if request_auth_github is not None else 7
request_hooks = dict(response=rate_limit(REQUEST_PERIOD))