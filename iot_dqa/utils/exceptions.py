class IoTDataQualityAssessmentBaseException(Exception):
    def __init__(self, message="An error occurred."):
        self.message = message
        super().__init__(self.message)


class InvalidFileException(IoTDataQualityAssessmentBaseException):
    """Raised when an invalid file is provided."""


class InvalidColumnMappingException(IoTDataQualityAssessmentBaseException):
    """Raised when an invalid column mapping is provided."""


class InvalidDimensionException(IoTDataQualityAssessmentBaseException):
    """Raised when an invalid dimension is provided."""


class InsufficientDataException(IoTDataQualityAssessmentBaseException):
    """Raised when the provided data is insufficient."""
