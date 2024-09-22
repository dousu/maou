from maou.interface import hcpe_converter_interface


class FileSystem(hcpe_converter_interface.FileLoader):
    @staticmethod
    def get_text(file_name: str, encoding: str = "utf-8") -> str:
        with open(file_name, encoding=encoding) as f:
            return f.read()
