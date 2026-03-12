from pydantic import BaseModel, ConfigDict, Field


class SubstitutionGroup(BaseModel):
    id: int
    code: str = Field(..., description="Short group code, e.g. 'burger_patty'.")
    name: str = Field(..., description="Human-readable group name, usually same as code.")
    description: str | None = Field(
        None,
        description="Optional longer text describing the group semantics.",
    )

    model_config = ConfigDict(from_attributes=True)

