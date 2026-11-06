"""Pipeline orchestrator module for Tasks 1, 2, and 3."""

from open_domain_qa.pipeline.runner import PipelineRunner
from open_domain_qa.pipeline.task1 import Task1Pipeline
from open_domain_qa.pipeline.task2 import Task2Pipeline
from open_domain_qa.pipeline.task3 import Task3Pipeline

__all__ = [
    "Task1Pipeline",
    "Task2Pipeline",
    "Task3Pipeline",
    "PipelineRunner",
]
