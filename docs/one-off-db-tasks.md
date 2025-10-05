# One-off DB tasks

- ssh into dokku host
- `dokku run aare-oraku-forecast`
- `python`
- Use the following. Either autocommit or conn.commit(), one is enough.

```python
import psycopg
import os

conn = psycopg.connect(os.getenv("ORAKU_CONNECTION_STRING"), autocommit=True)
cur = conn.execute("""...""")
cur.close()
conn.commit()
conn.close()
  ```

- Ctrl+D twice