class FleetCareError(Exception):
    """Base exception for FleetCare AI."""


class ResourceNotFoundError(FleetCareError):

    def __init__(
        self,
        resource: str,
        resource_id: object,
    ):
        self.resource = resource
        self.resource_id = resource_id

        super().__init__(
            f"{resource} '{resource_id}' was not found."
        )