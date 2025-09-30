from __future__ import annotations

from tenzir_test import run

stdout_lock = run.stdout_lock

update_registry_metadata = run.update_registry_metadata
get_allowed_extensions = run.get_allowed_extensions
get_test_env_and_config_args = run.get_test_env_and_config_args
parse_test_config = run.parse_test_config
report_failure = run.report_failure
handle_skip = run.handle_skip
run_simple_test = run.run_simple_test
print_diff = run.print_diff
success = run.success
fail = run.fail

__all__ = [
    "stdout_lock",
    "update_registry_metadata",
    "get_allowed_extensions",
    "get_test_env_and_config_args",
    "parse_test_config",
    "report_failure",
    "handle_skip",
    "run_simple_test",
    "print_diff",
    "success",
    "fail",
]
