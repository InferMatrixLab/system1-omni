"""Closed-loop replay with balanced complete manifest cycles and a final drain."""
import asyncio,time

async def collect(send, items, concurrency, duration=None, on_result=None, stop_on_failure=False):
    assert items and concurrency>0 and (duration is None or duration>0)
    index=0;stopped=False;rows=[];started=time.perf_counter()
    deadline=started+duration if duration else None
    async def worker():
        nonlocal index,stopped
        while not stopped:
            if deadline is None:
                if index>=len(items):return
            elif time.perf_counter()>=deadline and index%len(items)==0:
                stopped=True
                return
            ordinal=index;index+=1
            row=await send(items[ordinal%len(items)])
            row['dispatch_index']=ordinal
            rows.append(row)
            if on_result is not None:on_result(row)
            if stop_on_failure and (row.get('error') or row.get('state','completed')!='completed'):
                stopped=True
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return sorted(rows,key=lambda r:r['dispatch_index']),time.perf_counter()-started
