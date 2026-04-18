#!/bin/bash
set -e

INDEX_FILE="${RAGU_STORAGE:-ragu_data}/kv_chunks.json"

# Remove empty index files from failed builds
if [ -f "$INDEX_FILE" ] && [ "$(wc -c < "$INDEX_FILE")" -le 4 ]; then
    echo "[entrypoint] Removing empty index files from failed build..."
    rm -f "${RAGU_STORAGE:-ragu_data}"/*.json
fi

if [ ! -f "$INDEX_FILE" ]; then
    echo "[entrypoint] Index not found. Restoring DB + running ETL..."

    python -c "
import pymssql, os, time

host = os.getenv('MSSQL_HOST', 'mssql')
pwd  = os.getenv('MSSQL_SA_PASSWORD', 'BaltBeregHack2026!')

# Wait for SQL Server to accept connections
for i in range(90):
    try:
        c = pymssql.connect(server=host, user='SA', password=pwd)
        c.close()
        print(f'[restore] SQL Server ready after {i*2}s')
        break
    except:
        time.sleep(2)
else:
    print('[restore] ERROR: SQL Server did not start')
    exit(1)

# Check if DB exists
conn = pymssql.connect(server=host, user='SA', password=pwd)
cur = conn.cursor()
cur.execute(\"SELECT COUNT(*) FROM sys.databases WHERE name='service_desk_tdbb'\")
exists = cur.fetchone()[0]

if exists == 0:
    print('[restore] Restoring service_desk_tdbb...')
    conn.autocommit(True)
    cur.execute('''
        RESTORE DATABASE [service_desk_tdbb]
        FROM DISK = '/var/opt/mssql/backup/cleaned.bak'
        WITH MOVE 'IntraService3'                  TO '/var/opt/mssql/data/service_desk_tdbb.mdf',
             MOVE 'ftrow_TaskSearchingCatalog'     TO '/var/opt/mssql/data/service_desk_tdbb_search_cat.ndf',
             MOVE 'ftrow_KBDocumentFTC'            TO '/var/opt/mssql/data/service_desk_tdbb_document.ndf',
             MOVE 'ftrow_TaskParentSearchCatalog'  TO '/var/opt/mssql/data/service_desk_tdbb_parent_cat.ndf',
             MOVE 'IntraService3_log'              TO '/var/opt/mssql/data/service_desk_tdbb_log.ldf',
             REPLACE
    ''')
    print('[restore] Database restored!')
else:
    print('[restore] Database already exists')

conn.close()
"

    echo "[entrypoint] Running ETL..."
    python -m etl.extract_and_index
    echo "[entrypoint] ETL done."
else
    echo "[entrypoint] Index exists, skipping ETL."
fi

echo "[entrypoint] Starting API server..."
exec uvicorn server.main:app --host 0.0.0.0 --port 8000
