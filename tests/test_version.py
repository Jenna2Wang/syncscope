import re

import syncscope


def test_version_is_exposed():
    assert isinstance(syncscope.__version__, str)


def test_version_is_pep440_ish():
    # major.minor.patch with an optional pre-release suffix
    assert re.match(r"^\d+\.\d+\.\d+", syncscope.__version__)
