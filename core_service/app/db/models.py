import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class AgentSession(Base):
    __tablename__ = "agent_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4
    )
    status: Mapped[str] = mapped_column(
        String, nullable=False, default="pending"
    )  # pending / processing / complete / failed
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    car_image_urls: Mapped[list[str] | None] = mapped_column(
        ARRAY(String), nullable=True
    )
    events: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSONB, nullable=True
    )
    result_image_urls: Mapped[list[str] | None] = mapped_column(
        ARRAY(String), nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
