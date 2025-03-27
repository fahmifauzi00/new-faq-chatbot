import sys
import pysqlite3

# Override the sqlite3 module with pysqlite3
sys.modules["sqlite3"] = pysqlite3