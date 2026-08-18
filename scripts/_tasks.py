"""Task registration shared by the command-line entry points."""


def register_tasks() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

