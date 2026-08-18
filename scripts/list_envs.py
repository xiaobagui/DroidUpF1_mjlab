"""List built-in and locally registered tasks."""

from scripts._tasks import register_tasks


def main() -> None:
  register_tasks()
  from mjlab.scripts.list_envs import main as mjlab_main

  mjlab_main()


if __name__ == "__main__":
  main()

