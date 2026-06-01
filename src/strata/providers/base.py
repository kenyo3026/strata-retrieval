import pathlib



class DocAnalyzer:

    def __init__(self, source: pathlib.Path):
        self.source = source

    def analyze(self):
        ...