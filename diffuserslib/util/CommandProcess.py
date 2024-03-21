import asyncio


class CommandProcess:
    def __init__(self, command):
        self.command = command
        self.process = None


    def runSync(self):
        asyncio.run(self.run())


    async def run(self):
        self.stdout = ""
        self.stderr = ""
        if(isinstance(self.command, list)):
            command = " ".join(self.command)
        else:
            command = self.command
        print(f"Running command: {command}")
        self.process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        await asyncio.gather(
            self.read_stream(self.process.stdout, "stdout"),
            self.read_stream(self.process.stderr, "stderr")
        )
        await self.process.wait()
        print(f"Process exited with status: {self.process.returncode}")
        if self.process.returncode != 0:
            raise Exception(f"{self.stderr}")


    async def read_stream(self, stream, name):
        while True:
            line = await stream.readline()
            if not line:
                break
            print(f"{name}: {line.decode().strip()}")
            if(name == "stderr"):
                self.stderr += line.decode()
            else:
                self.stdout += line.decode()

