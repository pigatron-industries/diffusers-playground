from dataclasses import dataclass, field
import yaml
from typing import Dict


@dataclass
class DatabaseConnection:
    name:str
    connection:str
    exclude_tables:list = field(default_factory = lambda: [])
    include_tables:list = field(default_factory = lambda: [])


class DatabaseConnections:
    connections = {}

    @staticmethod
    def getDatabaseConnections() -> Dict[str, DatabaseConnection]:
        with open("./config/local_dbconfig.yml", 'r') as stream:
            data = yaml.safe_load(stream)
            for connection in data['connections']:
                DatabaseConnections.connections[connection['name']] = DatabaseConnection(
                    name = connection['name'], 
                    connection = connection['connection'], 
                    exclude_tables = connection.get('exclude_tables', []),
                    include_tables = connection.get('include_tables', []))
        return DatabaseConnections.connections