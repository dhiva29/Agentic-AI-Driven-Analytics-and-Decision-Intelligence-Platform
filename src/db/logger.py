from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config.settings import SETTINGS

Base = declarative_base()


class AgentLog(Base):
    __tablename__ = "agent_logs"

    id = Column(Integer, primary_key=True)
    agent_type = Column(String(120), nullable=False)
    input = Column(JSON, nullable=False)
    output = Column(JSON, nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class AuditLogger:
    def __init__(self) -> None:
        self.engine = create_engine(SETTINGS.db_url)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)

    def log(self, agent_type: str, input_payload: dict, output_payload: dict, confidence: float) -> None:
        session = self.session()
        try:
            record = AgentLog(
                agent_type=agent_type,
                input=json.loads(json.dumps(input_payload, default=str)),
                output=json.loads(json.dumps(output_payload, default=str)),
                confidence=confidence,
            )
            session.add(record)
            session.commit()
        finally:
            session.close()

    def fetch_recent(self, limit: int = 50) -> list[dict]:
        session = self.session()
        try:
            rows = session.query(AgentLog).order_by(AgentLog.timestamp.desc()).limit(limit).all()
            return [
                {
                    "agent_type": r.agent_type,
                    "input": r.input,
                    "output": r.output,
                    "confidence": r.confidence,
                    "timestamp": r.timestamp,
                }
                for r in rows
            ]
        finally:
            session.close()
