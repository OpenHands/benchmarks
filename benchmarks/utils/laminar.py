from __future__ import annotations

import json
import os
from typing import Any
from uuid import UUID

from lmnr import Laminar, LaminarClient
from pydantic import BaseModel

from openhands.sdk import get_logger


# Environment variables to forward to the workspace
LMNR_ENV_VARS = [
    "LMNR_PROJECT_API_KEY",
    "LMNR_SPAN_CONTEXT",
]

logger = get_logger(__name__)


class LaminarEvalMetadata(BaseModel):
    eval_id: UUID | None = None
    datapoint_id: UUID | None = None


class LaminarService:
    """Singleton helper around Laminar client usage."""

    _object: LaminarService | None = None

    def __init__(self) -> None:
        self._client: LaminarClient | None = None
        self._laminar_initialized = False

    @classmethod
    def get(cls) -> "LaminarService":
        if cls._object is None:
            cls._object = cls()
        return cls._object

    def _is_enabled(self) -> bool:
        return bool(os.environ.get("LMNR_PROJECT_API_KEY"))

    def initialize(self) -> bool:
        """
        Initialize the Laminar SDK once per process.
        Returns True if initialization succeeded (or was already done), False otherwise.
        """

        if self._laminar_initialized:
            return True

        if not self._is_enabled():
            return False

        try:
            Laminar.initialize()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Failed to initialize Laminar SDK: %s", exc)
            return False

        self._laminar_initialized = True
        return True

    def _get_client(self) -> LaminarClient | None:
        if not self._laminar_initialized or not self._is_enabled():
            return None

        if self._client is None:
            try:
                self._client = LaminarClient()
            except Exception as exc:
                logger.warning("Failed to create LaminarClient: %s", exc)
                return None

        return self._client

    def create_evaluation(
        self, name: str, group_name: str, metadata: dict[str, Any] | None = None
    ):
        client = self._get_client()
        if client is None:
            return None

        try:
            eval_id = client.evals.create_evaluation(
                name=name,
                group_name=group_name,
                metadata=metadata,
            )
            return eval_id
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(
                "Laminar evaluation %s (%s): %s",
                name,
                group_name,
                exc,
            )

    def create_evaluation_datapoint(
        self,
        eval_id: UUID | None,
        data: Any,
        metadata: dict[str, Any],
        index: int,
    ) -> tuple[UUID | None, str | None]:
        """
        Create a Laminar datapoint.
        Creates a new span for the evaluation and returns the span context.
        """

        if eval_id is None:
            return None, ""

        client = self._get_client()
        if client is None:
            return None, ""

        try:
            eval_span = Laminar.start_active_span(
                "Evaluation",
                span_type="EVALUATION",  # type: ignore
            )
            lmnr_span_ctx = Laminar.serialize_span_context(eval_span)
            eval_span.end()

            return client.evals.create_datapoint(
                eval_id=eval_id,
                data=data,
                target=1,
                metadata=metadata,
                index=index,
                trace_id=UUID(int=eval_span.get_span_context().trace_id),
            ), lmnr_span_ctx
        except Exception as exc:
            logger.debug(
                "Failed to create Laminar datapoint for eval %s: %s",
                eval_id,
                exc,
            )
            return None, None

    def _update_evaluation_datapoint(
        self,
        datapoint_id: UUID | None,
        eval_id: UUID | None,
        executor_output: Any,
        scores: dict[str, Any] = {},
    ) -> None:
        """
        Update a Laminar datapoint.
        """

        client = self._get_client()
        if client is None or not eval_id or not datapoint_id:
            return

        try:
            client.evals.update_datapoint(
                eval_id=eval_id,
                datapoint_id=datapoint_id,
                executor_output=executor_output,
                scores=scores,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(
                "Failed to update Laminar datapoint %s for eval %s: %s",
                datapoint_id,
                eval_id,
                exc,
            )

    def update_evaluation_datapoints_from_output_file(
        self,
        output_file: str,
        resolved_ids: set[str],
    ) -> None:
        """
        Update Laminar datapoints with scores based on an output.jsonl file.

        Reads the output file, extracts Laminar metadata (datapoint_id, eval_id)
        from each entry, and updates each datapoint with {"Score": 1} if the
        instance is in given resolved_ids, or {"Score": 0} otherwise.

        Args:
            output_file: Path to the output.jsonl file containing evaluation results
            resolved_ids: Set of instance IDs that are considered resolved/passed
        """
        if not self.initialize():
            logger.debug("Laminar not enabled, skipping score updates")
            return

        try:
            with open(output_file, "r") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        eval_output = json.loads(line)
                        instance_id = eval_output.get("instance_id")
                        metadata_dict = eval_output.get("metadata", {})

                        if not metadata_dict:
                            logger.debug(
                                f"Line {line_num}: No metadata for {instance_id}, skipping"
                            )
                            continue

                        # Extract Laminar metadata
                        lmnr_dict = metadata_dict.get("lmnr", {})
                        if not lmnr_dict:
                            logger.debug(
                                f"Line {line_num}: No lmnr metadata for {instance_id}, skipping"
                            )
                            continue

                        # Convert to LaminarEvalMetadata instance
                        try:
                            lmnr_metadata = LaminarEvalMetadata.model_validate(
                                lmnr_dict
                            )
                        except Exception as e:
                            logger.debug(
                                f"Line {line_num}: Failed to parse Laminar metadata for {instance_id}: {e}"
                            )
                            continue

                        if not lmnr_metadata.datapoint_id or not lmnr_metadata.eval_id:
                            logger.debug(
                                f"Line {line_num}: Missing Laminar IDs for {instance_id}, skipping"
                            )
                            continue

                        # Determine score: 1 if resolved, 0 otherwise
                        score = 1 if instance_id in resolved_ids else 0

                        # Update the Laminar datapoint with the score
                        self._update_evaluation_datapoint(
                            datapoint_id=lmnr_metadata.datapoint_id,
                            eval_id=lmnr_metadata.eval_id,
                            executor_output=eval_output,
                            scores={"Score": score},
                        )

                        logger.debug(f"Updated {instance_id}: Score={score}")
                    except json.JSONDecodeError as e:
                        logger.debug(f"Line {line_num}: Invalid JSON - {e}")
                    except Exception as e:
                        logger.debug(f"Line {line_num}: Error processing - {e}")

        except Exception as e:
            logger.debug(f"Failed to read output file: {e}")
            return

        logger.debug("Laminar score updates complete")
