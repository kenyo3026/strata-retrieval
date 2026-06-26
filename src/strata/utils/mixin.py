import datetime

class NameWithLazyDatetime:
    def __init__(
        self,
        prefix:str="lazy",
        suffix:str=None,
        datetime_format: str = "%Y-%m-%d_%H:%M:%S"
    ):
        self.prefix = prefix
        self.suffix = suffix
        self.datetime_format = datetime_format

    def __str__(self):
        name = []

        if self.prefix:
            name.append(self.prefix)

        name.append(datetime.datetime.now().strftime(self.datetime_format))

        if self.suffix:
            name.append(self.suffix)

        return '_'.join(name)

    def __call__(self):
        return str(self)