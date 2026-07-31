from __future__ import annotations


class SCAPSParserBaseException(Exception):
    """Base exception for all SCAPS parser errors."""
    pass


class UnsupportedFormatException(SCAPSParserBaseException):
    """Raised when file format or extension is not supported by any registered reader."""
    pass


class FileCorruptedException(SCAPSParserBaseException):
    """Raised when file content is corrupted or unreadable."""
    pass


class SegmentationException(SCAPSParserBaseException):
    """Raised when layout discovery or table segmentation fails."""
    pass


class DataCorruptionException(SCAPSParserBaseException):
    """Raised when extracted data violates structural integrity."""
    pass
