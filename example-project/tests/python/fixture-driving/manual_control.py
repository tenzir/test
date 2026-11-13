#!/usr/bin/env python
# timeout: 20

import signal

# The harness registers the fixtures of this example project in the test
# process, so the test acquires them without importing them.

# Context-manager style: `with` calls start() on enter and stop() on exit.
with acquire_fixture("server") as server:
    print("# server started via context manager")
    server.kill(signal.SIGTERM)
    print("# sent SIGTERM to server")

print("# server stopped automatically on context exit")

# Manual style: same controller API, but we start/stop explicitly ourselves, as
# opposed to letting `with` do it for us.
server = acquire_fixture("server")
server.start()
print("# server started manually")
server.stop()
print("# server stopped manually")
