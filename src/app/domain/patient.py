from pydantic import BaseModel, ConfigDict, Field


class PatientRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(..., min_length=1)
    name: str | None = None
    age: int | None = Field(default=None, ge=0, le=150)
