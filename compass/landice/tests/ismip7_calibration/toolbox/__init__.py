"""
The ISMIP7 parameter-selection toolbox, vendored from upstream.

:py:mod:`~compass.landice.tests.ismip7_calibration.toolbox.parameter_selection_toolbox`
is a verbatim copy of ``parameterisations/parameter_selection_toolbox.py`` from
https://github.com/ismip/ismip7-antarctic-ocean-forcing.  It implements the
objective function of the ISMIP7 AIS ice-ocean protocol (Reese et al.,
Sect. 4.2) and the downstream ``deltaT`` fit.  ``PROVENANCE.md`` in this
directory records exactly which upstream revision it came from and how to
update it.

The copy must stay byte-for-byte identical to upstream, so that a future
refresh is a clean replace and so that the published calibration numbers
cannot change without anyone noticing.  :py:func:`check_integrity` enforces
that, and is called on import.
"""  # noqa: E501

import hashlib
import os

#: SHA256 of the vendored toolbox; see ``PROVENANCE.md``
EXPECTED_SHA256 = \
    '3d016c04987d7c66c2603e2c3110d8a6a68cee0b6193c7f47bccf64aaa1ce029'

#: upstream commit that last modified the vendored file
UPSTREAM_COMMIT = '132beb155e63fd07957e28407a83cb9d2c954447'

#: upstream repository the file was copied from
UPSTREAM_URL = 'https://github.com/ismip/ismip7-antarctic-ocean-forcing'


def toolbox_path():
    """
    The path to the vendored toolbox source file.

    Returns
    -------
    path : str
        Absolute path to ``parameter_selection_toolbox.py``
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'parameter_selection_toolbox.py')


def file_sha256():
    """
    The SHA256 of the vendored toolbox source file, as it is on disk.

    Returns
    -------
    digest : str
        Hex digest of ``parameter_selection_toolbox.py``
    """
    with open(toolbox_path(), 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def check_integrity():
    """
    Verify that the vendored toolbox still matches the recorded upstream copy.

    Raises
    ------
    RuntimeError
        If the file on disk does not match :py:data:`EXPECTED_SHA256`.  The
        vendored copy must be an unmodified upstream file, because the
        published ISMIP7 calibration numbers depend on it; a local edit or a
        partial update should fail loudly rather than quietly change results.
    """
    digest = file_sha256()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            f'The vendored ISMIP7 parameter-selection toolbox at\n'
            f'  {toolbox_path()}\n'
            f'does not match the recorded upstream copy.\n'
            f'  expected SHA256: {EXPECTED_SHA256}\n'
            f'  found SHA256:    {digest}\n'
            f'The file must be a verbatim copy of\n'
            f'  {UPSTREAM_URL}\n'
            f'  parameterisations/parameter_selection_toolbox.py\n'
            f'at commit {UPSTREAM_COMMIT}.  If you are deliberately updating '
            f'it, follow the instructions in PROVENANCE.md in this '
            f'directory.')


check_integrity()

from compass.landice.tests.ismip7_calibration.toolbox import (  # noqa: E402
    parameter_selection_toolbox,
)

__all__ = ['parameter_selection_toolbox', 'check_integrity', 'file_sha256',
           'toolbox_path', 'EXPECTED_SHA256', 'UPSTREAM_COMMIT',
           'UPSTREAM_URL']
