import asyncio


class CommandProcess:
    def __init__(self, command):
        self.command = command


    def runSync(self):
        asyncio.run(self.run())


    async def run(self):
        print(f"Starting process: {self.command}")
        process = await asyncio.create_subprocess_shell(
            self.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        await asyncio.gather(
            self.read_stream(process.stdout, "stdout"),
            self.read_stream(process.stderr, "stderr")
        )


    async def read_stream(self, stream, name):
        while True:
            line = await stream.readline()
            if not line:
                break
            print(f"{name}: {line.decode().strip()}")

