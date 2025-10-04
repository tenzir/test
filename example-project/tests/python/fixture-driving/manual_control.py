#!/usr/bin/env python
# timeout: 20

import signal

# Importing registers the fixtures defined in this example project as a side
# effect.
import fixtures

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
