class Error(Exception):
    """Base class for other exceptions"""
    pass


class InvalidFillNAError(Error):
    """Invalid arguments when using the fill_na() method."""
    pass


class InvalidFilenameError(Error):
    """Invalid filename used."""
    pass


class InvalidOrderByStrategy(Error):
    """Invalid value of orderBy_strategy."""
    pass
